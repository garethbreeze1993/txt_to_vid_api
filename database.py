import os
import sqlite3
import threading
from contextlib import contextmanager
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta



from dotenv import load_dotenv
load_dotenv()  # take environment variables

DB_PATH = os.getenv("DB_PATH")
JOB_EXPIRATION_DAYS = 7

db_lock = threading.Lock()


class JobManager:
    """Handles job operations with SQLite"""

    @staticmethod
    def init_db():
        """Initialize the database with job table"""
        with db_lock:
            conn = sqlite3.connect(DB_PATH)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    message TEXT,
                    prompt TEXT NOT NULL,
                    video_id TEXT NOT NULL,
                    video_url TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    expires_at TEXT NOT NULL
                )
            ''')

            # Create index for cleanup queries
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_expires_at ON jobs(expires_at)
            ''')

            # Create index for status queries
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_status ON jobs(status)
            ''')

            conn.commit()
            conn.close()

    @staticmethod
    @contextmanager
    def get_db():
        """Context manager for database connections"""
        with db_lock:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            try:
                yield conn
            finally:
                conn.close()

    @staticmethod
    def create_job(job_id: str, prompt: str, video_id: int) -> Dict:
        """Create a new job entry"""
        now = datetime.now()
        expires_at = now + timedelta(days=JOB_EXPIRATION_DAYS)

        job_data = {
            "job_id": job_id,
            "status": "pending",
            "message": "Video generation started",
            "prompt": prompt,
            "video_id": video_id,
            "video_url": None,
            "created_at": now.isoformat(),
            "completed_at": None,
            "expires_at": expires_at.isoformat()
        }

        with JobManager.get_db() as conn:
            conn.execute('''
                INSERT INTO jobs (job_id, status, message, prompt, video_id, 
                                video_url, created_at, completed_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                job_data["job_id"], job_data["status"], job_data["message"],
                job_data["prompt"], job_data["video_id"], job_data["video_url"],
                job_data["created_at"], job_data["completed_at"], job_data["expires_at"]
            ))
            conn.commit()

        return job_data

    @staticmethod
    def get_job(job_id: str) -> Optional[Dict]:
        """Get job data by ID"""
        with JobManager.get_db() as conn:
            cursor = conn.execute('''
                SELECT * FROM jobs WHERE job_id = ? AND expires_at > ?
            ''', (job_id, datetime.now().isoformat()))

            row = cursor.fetchone()
            if row:
                return dict(row)
        return None

    @staticmethod
    def update_job(job_id: str, updates: Dict) -> bool:
        """Update job with new data"""
        if not updates:
            return False

        # Build dynamic UPDATE query
        set_clauses = []
        values = []

        for key, value in updates.items():
            if key in ['status', 'message', 'video_url', 'completed_at']:
                set_clauses.append(f"{key} = ?")
                values.append(value)

        if not set_clauses:
            return False

        values.append(job_id)

        with JobManager.get_db() as conn:
            cursor = conn.execute(f'''
                UPDATE jobs SET {', '.join(set_clauses)} 
                WHERE job_id = ? AND expires_at > ?
            ''', values + [datetime.now().isoformat()])

            conn.commit()
            return cursor.rowcount > 0

    @staticmethod
    def delete_job(job_id: str) -> bool:
        """Delete a job"""
        with JobManager.get_db() as conn:
            cursor = conn.execute('DELETE FROM jobs WHERE job_id = ?', (job_id,))
            conn.commit()
            return cursor.rowcount > 0

    @staticmethod
    def list_jobs(limit: int = 50, status: Optional[str] = None) -> List[Dict]:
        """List jobs with optional status filter"""
        with JobManager.get_db() as conn:
            if status:
                cursor = conn.execute('''
                    SELECT * FROM jobs 
                    WHERE status = ? AND expires_at > ?
                    ORDER BY created_at DESC 
                    LIMIT ?
                ''', (status, datetime.now().isoformat(), limit))
            else:
                cursor = conn.execute('''
                    SELECT * FROM jobs 
                    WHERE expires_at > ?
                    ORDER BY created_at DESC 
                    LIMIT ?
                ''', (datetime.now().isoformat(), limit))

            return [dict(row) for row in cursor.fetchall()]

    @staticmethod
    def cleanup_expired_jobs() -> int:
        """Remove expired jobs"""
        with JobManager.get_db() as conn:
            cursor = conn.execute('''
                DELETE FROM jobs WHERE expires_at <= ?
            ''', (datetime.now().isoformat(),))
            conn.commit()
            return cursor.rowcount
CREATE TABLE IF NOT EXISTS evaluations (
    evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_text TEXT,
    question TEXT,
    original_score REAL,
    corrected_score REAL,
    rag_reference_ids TEXT,
    policy_version TEXT,
    teacher_action TEXT CHECK(teacher_action IN ('accepted', 'corrected')),
    reward REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_id INTEGER,
    feedback_text TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (evaluation_id) REFERENCES evaluations(evaluation_id)
);

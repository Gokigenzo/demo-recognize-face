-- Enable vector extension for pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Students Table
CREATE TABLE IF NOT EXISTS students (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id VARCHAR(255) UNIQUE NOT NULL, -- original user_id slug (e.g. 'ada_lovelace')
    name VARCHAR(255) NOT NULL,
    avatar_url TEXT,
    registration_date TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    active BOOLEAN NOT NULL DEFAULT true
);

-- 2. Embeddings Table
CREATE TABLE IF NOT EXISTS embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id UUID NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    embedding_vector vector(512) NOT NULL,
    pose VARCHAR(50),
    quality_score DOUBLE PRECISION,
    created_time TIMESTAMPTZ NOT NULL DEFAULT now(),
    embedding_source VARCHAR(100)
);

-- 3. Trained Classifier Metadata Table
CREATE TABLE IF NOT EXISTS trained_classifiers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(255) NOT NULL,
    training_date TIMESTAMPTZ NOT NULL DEFAULT now(),
    training_accuracy DOUBLE PRECISION,
    precision DOUBLE PRECISION,
    recall DOUBLE PRECISION,
    f1 DOUBLE PRECISION,
    hyperparameters JSONB,
    model_version VARCHAR(50),
    model_path TEXT, -- path/URL in Supabase Storage
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 4. Attendance Table
CREATE TABLE IF NOT EXISTS attendance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id UUID REFERENCES students(id) ON DELETE CASCADE, -- NULL if Unknown
    timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),
    confidence DOUBLE PRECISION NOT NULL,
    recognition_method VARCHAR(100),
    status VARCHAR(50) DEFAULT 'present',
    session_id UUID,
    duplicate_detected BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 5. Monitoring Feedback Table
CREATE TABLE IF NOT EXISTS monitoring_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id UUID REFERENCES students(id) ON DELETE SET NULL, -- student uuid matching correct student
    predicted_name VARCHAR(255),
    correct_name VARCHAR(255),
    correct_student_id VARCHAR(255), -- slug student id
    confidence DOUBLE PRECISION,
    user_decision VARCHAR(100),
    note TEXT,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 6. Session Statistics Table
CREATE TABLE IF NOT EXISTS session_statistics (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ,
    present_count INTEGER NOT NULL DEFAULT 0,
    absent_count INTEGER NOT NULL DEFAULT 0,
    unknown_count INTEGER NOT NULL DEFAULT 0,
    average_confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    average_fps DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    recognition_time DOUBLE PRECISION NOT NULL DEFAULT 0.0, -- in ms
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 7. Application Configuration Table
CREATE TABLE IF NOT EXISTS application_configuration (
    id VARCHAR(50) PRIMARY KEY DEFAULT 'default',
    recognition_threshold DOUBLE PRECISION NOT NULL DEFAULT 0.45,
    confirmation_frames INTEGER NOT NULL DEFAULT 5,
    detection_interval INTEGER NOT NULL DEFAULT 2,
    camera_resolution_width INTEGER NOT NULL DEFAULT 640,
    camera_resolution_height INTEGER NOT NULL DEFAULT 480,
    recognition_fps DOUBLE PRECISION NOT NULL DEFAULT 5.0,
    ui_preferences JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Enable RLS on all tables
ALTER TABLE students ENABLE ROW LEVEL SECURITY;
ALTER TABLE embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE trained_classifiers ENABLE ROW LEVEL SECURITY;
ALTER TABLE attendance ENABLE ROW LEVEL SECURITY;
ALTER TABLE monitoring_feedback ENABLE ROW LEVEL SECURITY;
ALTER TABLE session_statistics ENABLE ROW LEVEL SECURITY;
ALTER TABLE application_configuration ENABLE ROW LEVEL SECURITY;

-- Add permissive RLS policies for simple demo access
CREATE POLICY "Allow public access" ON students FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow public access" ON embeddings FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow public access" ON trained_classifiers FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow public access" ON attendance FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow public access" ON monitoring_feedback FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow public access" ON session_statistics FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow public access" ON application_configuration FOR ALL USING (true) WITH CHECK (true);

-- Indexes for query optimization
CREATE INDEX IF NOT EXISTS idx_embeddings_student ON embeddings(student_id);
CREATE INDEX IF NOT EXISTS idx_attendance_student ON attendance(student_id);
CREATE INDEX IF NOT EXISTS idx_attendance_session ON attendance(session_id);
CREATE INDEX IF NOT EXISTS idx_attendance_time ON attendance(timestamp);
CREATE INDEX IF NOT EXISTS idx_feedback_student ON monitoring_feedback(student_id);

-- 8. Supabase Storage RLS Policies (For 'models' and 'attendance' buckets)
-- These allow the client-side anonymous key to upload and access pickles & CSV reports.
CREATE POLICY "Allow public upload to models" ON storage.objects FOR INSERT TO anon, authenticated WITH CHECK (bucket_id = 'models');
CREATE POLICY "Allow public select from models" ON storage.objects FOR SELECT TO anon, authenticated USING (bucket_id = 'models');
CREATE POLICY "Allow public update of models" ON storage.objects FOR UPDATE TO anon, authenticated USING (bucket_id = 'models') WITH CHECK (bucket_id = 'models');

CREATE POLICY "Allow public upload to attendance" ON storage.objects FOR INSERT TO anon, authenticated WITH CHECK (bucket_id = 'attendance');
CREATE POLICY "Allow public select from attendance" ON storage.objects FOR SELECT TO anon, authenticated USING (bucket_id = 'attendance');
CREATE POLICY "Allow public update of attendance" ON storage.objects FOR UPDATE TO anon, authenticated USING (bucket_id = 'attendance') WITH CHECK (bucket_id = 'attendance');

"""Repository package for Supabase-backed persistence.

Exposes domain repositories and the shared connection manager so the rest
of the application can import them conveniently::

    from repositories import get_manager, StudentRepository
"""
from __future__ import annotations

from repositories.base_repository import SupabaseManager, get_manager
from repositories.student_repository import StudentRepository, get_student_repo
from repositories.embedding_repository import EmbeddingRepository, get_embedding_repo
from repositories.attendance_repository import AttendanceRepository, get_attendance_repo
from repositories.feedback_repository import FeedbackRepository, get_feedback_repo
from repositories.statistics_repository import StatisticsRepository, get_statistics_repo
from repositories.configuration_repository import ConfigurationRepository, get_configuration_repo
from repositories.classifier_repository import ClassifierRepository, get_classifier_repo

__all__ = [
    "SupabaseManager",
    "get_manager",
    "StudentRepository",
    "get_student_repo",
    "EmbeddingRepository",
    "get_embedding_repo",
    "AttendanceRepository",
    "get_attendance_repo",
    "FeedbackRepository",
    "get_feedback_repo",
    "StatisticsRepository",
    "get_statistics_repo",
    "ConfigurationRepository",
    "get_configuration_repo",
    "ClassifierRepository",
    "get_classifier_repo",
]


from django.test import TestCase
from .models import Document, Query


class DocumentTestCase(TestCase):
    def setUp(self):
        Document.objects.create(
            title="Test Document",
            content="This is a test document"
        )

    def test_document_creation(self):
        doc = Document.objects.get(title="Test Document")
        self.assertEqual(doc.content, "This is a test document")


class QueryTestCase(TestCase):
    def setUp(self):
        Query.objects.create(
            question="What is RAG?",
            answer="RAG is Retrieval-Augmented Generation"
        )

    def test_query_creation(self):
        query = Query.objects.get(question="What is RAG?")
        self.assertEqual(query.answer, "RAG is Retrieval-Augmented Generation")

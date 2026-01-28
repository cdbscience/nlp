from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="rag_index"),
    path("query/", views.query_api, name="rag_query"),
]

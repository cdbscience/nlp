from django.contrib import admin
from django import forms
from django.core.files.uploadedfile import UploadedFile
from .models import File, Document, Query
import json


class PDFUploadForm(forms.ModelForm):
    """Custom form for uploading PDF files"""
    pdf_file = forms.FileField(
        label='PDF File',
        help_text='Upload a PDF file (max 50MB)',
        required=True
    )

    class Meta:
        model = File
        fields = ['name']
        widgets = {
            'name': forms.TextInput(attrs={
                'placeholder': 'Enter a name for this file (optional)',
                'class': 'vTextField'
            })
        }

    def clean_pdf_file(self):
        file = self.files.get('pdf_file')
        if file:
            if not file.name.lower().endswith('.pdf'):
                raise forms.ValidationError('Only PDF files are allowed')
            if file.size > 50 * 1024 * 1024:  # 50MB limit
                raise forms.ValidationError('File size must be less than 50MB')
        return file


@admin.register(File)
class FileAdmin(admin.ModelAdmin):
    """Admin interface for PDF file uploads"""
    list_display = ('name', 'total_documents', 'created_at', 'file_size')
    search_fields = ('name',)
    list_filter = ('created_at',)
    readonly_fields = ('created_at', 'updated_at', 'total_documents', 'file_size_display', 'file_path', 'file_size')

    def get_fieldsets(self, request, obj=None):
        """Different fieldsets for add vs change views"""
        if obj is None:  # Adding new file
            return (
                ('Upload PDF', {
                    'fields': ('name', 'pdf_file'),
                    'description': 'Upload a PDF file and give it a name'
                }),
            )
        else:  # Editing existing file
            return (
                ('File Information', {
                    'fields': ('name', 'file_path', 'created_at', 'updated_at')
                }),
                ('Statistics', {
                    'fields': ('total_documents', 'file_size_display'),
                    'classes': ('collapse',)
                }),
            )

    def file_size_display(self, obj):
        """Display file size in human readable format"""
        try:
            size = obj.file_size / 1024  # Convert to KB
            if size > 1024:
                return f"{size / 1024:.2f} MB"
            return f"{size:.2f} KB"
        except:
            return "Unknown"
    file_size_display.short_description = "File Size"

    def get_form(self, request, obj=None, **kwargs):
        """Use custom form with file upload for add view"""
        if obj is None:  # Adding new file
            return PDFUploadForm
        return super().get_form(request, obj, **kwargs)

    def save_model(self, request, obj, form, change):
        """Process PDF file when saving"""
        if not change:  # Adding new file
            from .views import process_pdf_file
            from django.contrib import messages
            
            pdf_file = form.cleaned_data.get('pdf_file')
            
            if pdf_file:
                try:
                    # Process the PDF file using the shared function
                    result = process_pdf_file(pdf_file)
                    
                    if result.get('status') == 'success':
                        # File was processed successfully
                        obj.file_path = pdf_file.name
                        obj.total_documents = result.get('document_count', 0)
                        obj.is_processed = True
                        super().save_model(request, obj, form, change)
                        messages.success(request, f'✅ PDF "{obj.name}" uploaded and processed successfully with {obj.total_documents} documents')
                    else:
                        error_msg = result.get('error', 'Unknown error processing PDF')
                        messages.error(request, f'❌ Error processing PDF: {error_msg}')
                        raise Exception(error_msg)
                except Exception as e:
                    messages.error(request, f'❌ Error uploading PDF: {str(e)}')
                    raise
        else:
            super().save_model(request, obj, form, change)

    def has_delete_permission(self, request, obj=None):
        """Only staff can delete"""
        return request.user.is_staff


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    """Admin interface for documents"""
    list_display = ('title', 'file_name', 'created_at', 'preview')
    search_fields = ('title', 'content')
    list_filter = ('created_at', 'file')
    readonly_fields = ('created_at', 'updated_at', 'content_preview', 'embedding_json', 'chunk_index', 'word_count', 'char_count')
    
    fieldsets = (
        ('Content', {
            'fields': ('file', 'title', 'content', 'content_preview', 'chunk_index')
        }),
        ('Statistics', {
            'fields': ('word_count', 'char_count'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('metadata', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
        ('Embeddings', {
            'fields': ('embedding_json',),
            'classes': ('collapse',)
        }),
    )

    def file_name(self, obj):
        """Display associated file name"""
        return obj.file.name if obj.file else "No file"
    file_name.short_description = "File"

    def preview(self, obj):
        """Show content preview"""
        preview = obj.content[:100] if obj.content else "No content"
        return f"{preview}..."
    preview.short_description = "Preview"

    def content_preview(self, obj):
        """Read-only content preview"""
        return obj.content
    content_preview.short_description = "Full Content"

    def has_delete_permission(self, request, obj=None):
        """Only staff can delete"""
        return request.user.is_staff

    def has_add_permission(self, request):
        """Documents are created via PDF upload"""
        return False

    def has_delete_permission(self, request, obj=None):
        """Only staff can delete"""
        return request.user.is_staff


@admin.register(Query)
class QueryAdmin(admin.ModelAdmin):
    """Admin interface for queries"""
    list_display = ('question', 'created_at', 'has_answer')
    search_fields = ('question', 'answer')
    list_filter = ('created_at',)
    readonly_fields = ('question', 'answer', 'created_at')

    def has_add_permission(self, request):
        """Queries are created via the RAG interface"""
        return False

    def has_delete_permission(self, request, obj=None):
        """Only staff can delete"""
        return request.user.is_staff

    def has_answer(self, obj):
        """Check if query has an answer"""
        return bool(obj.answer)
    has_answer.boolean = True
    has_answer.short_description = "Has Answer"

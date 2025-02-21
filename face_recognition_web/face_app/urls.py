from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('live-detection/', views.live_detection, name='live_detection'),
    path('process-frame/', views.process_frame, name='process_frame'),
    path('upload-image/', views.upload_image, name='upload_image'),
]
"""myproject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from app1 import views as v1

urlpatterns = [
    path('admin/', admin.site.urls),
    path('adr1/',v1.Hello),
    path('run/',v1.run),
    path('mask/',v1.mask),
    path('about/',v1.about),
    path('abt/',v1.abt),
    path('run1/',v1.run1,name='run1'),
    path('mask1/',v1.mask1,name='mask1'),
]

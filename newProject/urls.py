"""newProject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
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
from django.urls import path,include
from django.conf.urls import url
from django.conf.urls.static import static
from django.conf import settings
from a import views

urlpatterns = [

    path('admin/', admin.site.urls),
    url(r'^$',views.renderPage,name='renderPage'),
    url(r'^firstLoad/$',views.index,name='index'),
    url(r'^firstStep/$',views.firstStep_1,name='firstStep'),
    url(r'^lastStep/$',views.firstStep_2,name='lastStep'),
    url(r'^finish/$',views.firstStep_3,name='firstStep_3'),
    # mobile roots
    url(r'^submit1/$',views.DBViewSet.submit1,name='submit1'),
    url(r'^submit2/$',views.DBViewSet.submit2,name='submit2'),
    url(r'^start/$',views.DBViewSet.start,name='start'),
    url(r'^stop/$',views.DBViewSet.stop_mobile,name='stop_mobile'),

]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


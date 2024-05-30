from django.urls import path
from .views import index,second,video_feed,test, firework

from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('', index, name='index'),    #홈 화면 url
    path('video_feed/', video_feed, name='video_feed'),  # 인공지능 화면 url
    path('test/', test, name='test'),   #jsonresponse 페이지
    path('second/', second, name='second'),    #아바타 뷰 url
    path('firework/', firework, name='firework'),
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)


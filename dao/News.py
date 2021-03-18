from sqlalchemy import Column,String,JSON,DateTime
from dao import Base

class News(Base):
    __tablename__ = 'news'
    article_id = Column(String(20),primary_key=True)
    media = Column(String(20))
    url = Column(String(100))
    title = Column(String(50))
    tags = Column(JSON)
    content = Column(String(20000))
    publish_time = Column(DateTime)
    images = Column(JSON)

    def __init__(self,article_id,media,url,title,tags,content,publish_time,images):
        self.article_id = article_id
        self.media = media
        self.url = url
        self.title = title
        self.tags = tags
        self.content = content
        self.publish_time = publish_time
        self.images = images
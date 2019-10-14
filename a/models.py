from django.db import models
from django.utils import timezone
import datetime
from datetime import date


def save_orginalpic(instance, filename):

    return "orginal/{filename.split('.')[0]}.{filename.split('.')[1]}"

# def save_heatmap(instance, filename):
#
#     return f"heatmap/{filename.split('.')[0]}.{filename.split('.')[1]}"

# Create your models here.
class Models(models.Model):

    name=models.CharField(max_length=200)

    class Meta:
        ordering=('name',)


    def __str__(self):

        return str(self.name)

class Sample(models.Model):

    created=models.DateField(auto_now_add=True)
    orgImg=models.ImageField(upload_to='orginal/',default=None, blank=True, null=True)
    heatMap=models.CharField(max_length=1000,default=None, blank=True, null=True)
    withModel=models.ForeignKey(Models, on_delete=models.CASCADE,default=None, blank=True, null=True)
    predict = models.CharField(max_length=10,default='true',blank=True, null=True)

    class Meta:
        ordering = ('created',)

    def __str__(self):
        return str(self.withModel)



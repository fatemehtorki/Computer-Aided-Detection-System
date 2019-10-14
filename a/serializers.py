from rest_framework import  serializers
from .models import Sample,Models



class SampleSerializer(serializers.ModelSerializer):
    withModel = serializers.CharField(source='withModel.name')
    class Meta:

        model = Sample
        fields = ('id','created','withModel','predict','orgImg','heatMap')

class ModelsSerializer(serializers.ModelSerializer):

    class Meta:

        model = Models
        fields = '__all__'

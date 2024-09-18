from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from nltk import word_tokenize
from nltk.stem import LancasterStemmer

def stem_text(text):
    tokens = word_tokenize(text)
    stemmer = LancasterStemmer()
    
    res = [stemmer.stem(token) for token in tokens]

    return " ".join(res)

@csrf_exempt
def index(request):
    result = ''
    if request.method == 'POST':
        input_text = request.POST.get('sentence', '')
        result = stem_text(input_text)
        return JsonResponse({'result': result})

    return render(request, 'stemming.html')

from django.shortcuts import render

def levenshtein_distance(word1, word2):
    rows = len(word1) + 1
    cols = len(word2) + 1

    # Initialize the distance matrix
    distance_matrix = [[0 for _ in range(cols)] for _ in range(rows)]

    # Set up the headers in the first row and column
    for i in range(1, rows):
        distance_matrix[i][0] = i  # Row headers for the leftmost column
    for j in range(1, cols):
        distance_matrix[0][j] = j  # Column headers for the top row

    # Compute the distance matrix
    for i in range(1, rows):
        for j in range(1, cols):
            if word1[i - 1] == word2[j - 1]:
                cost = 0
            else:
                cost = 1

            distance_matrix[i][j] = min(
                distance_matrix[i - 1][j] + 1,       # Deletion
                distance_matrix[i][j - 1] + 1,       # Insertion
                distance_matrix[i - 1][j - 1] + cost # Substitution
            )

    # Convert the matrix to a format suitable for display
    display_matrix = [[' ']*(cols+1) for _ in range(rows)]
    for i in range(rows):
        for j in range(cols+1):
            if(i==0 and j==0):
                display_matrix[i][j] = {"color": "green","value": ' '}
            elif(j == 0):
                display_matrix[i][j] = {"color": "green","value": word1[i-1]}
            else:
                display_matrix[i][j] = {"color": "","value": distance_matrix[i][j-1]}

    return {
        "matrix": display_matrix,
        "distance": distance_matrix[-1][-1],
        "word1": word1,
        "word2": word2,
    }


def index(request):
    result = {}
    if request.method == 'POST':
        
        word1 = request.POST.get('word1', '')
        word2 = request.POST.get('word2', '')
        
        result = levenshtein_distance(word1, word2)

    return render(request, 'levenshtein.html', {'result': result})

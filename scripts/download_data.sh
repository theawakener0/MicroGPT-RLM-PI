#!/bin/bash
# Download and clean training data for MicroGPT-RLM
# Pure bash/curl - NO Python required!

mkdir -p data

echo "=== MicroGPT-RLM-PI Data Download Script ==="
echo ""

# Check if curl is available
if ! command -v curl &> /dev/null; then
    echo "Error: curl not found"
    exit 1
fi

# Helper function to clean Gutenberg text
clean_gutenberg() {
    local input=$1
    local output=$2
    
    # Remove Gutenberg header (before "*** START OF")
    sed -n '/\*\*\* START OF/,$p' "$input" > "$output.tmp"
    
    # Remove Gutenberg footer (after "*** END OF")  
    sed -n '1,/\*\*\* END OF/p' "$output.tmp" > "$output.tmp2"
    
    # Remove lines starting with "Chapter" (optional, keeps structure)
    # Remove empty lines
    grep -v '^[[:space:]]*$' "$output.tmp2" > "$output"
    
    # Cleanup temp files
    rm -f "$output.tmp" "$output.tmp2"
}

echo "Downloading and cleaning datasets..."
echo ""

# ============================================
# 1. NAMES (keep separate - good for name generation)
# ============================================
if [ ! -f "data/names.txt" ]; then
    echo "1. Downloading names..."
    curl -L -o data/names.txt \
        "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt" \
        --silent --show-error
    echo "   Done! ($(wc -l < data/names.txt) names)"
else
    echo "1. names.txt already exists"
fi
echo ""

# ============================================
# 2. PRIDE AND PREJUDICE (Gutenberg ID 1342)
# ============================================
if [ ! -f "data/pride_prejudice.txt" ]; then
    echo "2. Downloading Pride and Prejudice..."
    curl -L -o data/pride_prejudice.txt \
        "https://www.gutenberg.org/files/1342/1342-0.txt" \
        --silent --show-error
    clean_gutenberg data/pride_prejudice.txt data/pride_prejudice_clean.txt
    mv data/pride_prejudice_clean.txt data/pride_prejudice.txt
    echo "   Done! ($(wc -l < data/pride_prejudice.txt) lines)"
else
    echo "2. pride_prejudice.txt already exists"
fi
echo ""

# ============================================
# 3. FRANKENSTEIN (Gutenberg ID 84)
# ============================================
if [ ! -f "data/frankenstein.txt" ]; then
    echo "3. Downloading Frankenstein..."
    curl -L -o data/frankenstein.txt \
        "https://www.gutenberg.org/files/84/84-0.txt" \
        --silent --show-error
    clean_gutenberg data/frankenstein.txt data/frankenstein_clean.txt
    mv data/frankenstein_clean.txt data/frankenstein.txt
    echo "   Done! ($(wc -l < data/frankenstein.txt) lines)"
else
    echo "3. frankenstein.txt already exists"
fi
echo ""

# ============================================
# 4. MOBY DICK (Gutenberg ID 2701)
# ============================================
if [ ! -f "data/moby_dick.txt" ]; then
    echo "4. Downloading Moby Dick..."
    curl -L -o data/moby_dick.txt \
        "https://www.gutenberg.org/files/2701/2701-0.txt" \
        --silent --show-error
    clean_gutenberg data/moby_dick.txt data/moby_dick_clean.txt
    mv data/moby_dick_clean.txt data/moby_dick.txt
    echo "   Done! ($(wc -l < data/moby_dick.txt) lines)"
else
    echo "4. moby_dick.txt already exists"
fi
echo ""

# ============================================
# 5. SHERLOCK HOLMES (Gutenberg ID 1661)
# ============================================
if [ ! -f "data/sherlock_holmes.txt" ]; then
    echo "5. Downloading Sherlock Holmes..."
    curl -L -o data/sherlock_holmes.txt \
        "https://www.gutenberg.org/files/1661/1661-0.txt" \
        --silent --show-error
    clean_gutenberg data/sherlock_holmes.txt data/sherlock_holmes_clean.txt
    mv data/sherlock_holmes_clean.txt data/sherlock_holmes.txt
    echo "   Done! ($(wc -l < data/sherlock_holmes.txt) lines)"
else
    echo "5. sherlock_holmes.txt already exists"
fi
echo ""

# ============================================
# 6. ALICE IN WONDERLAND (Gutenberg ID 11)
# ============================================
if [ ! -f "data/alice_wonderland.txt" ]; then
    echo "6. Downloading Alice in Wonderland..."
    curl -L -o data/alice_wonderland.txt \
        "https://www.gutenberg.org/files/11/11-0.txt" \
        --silent --show-error
    clean_gutenberg data/alice_wonderland.txt data/alice_wonderland_clean.txt
    mv data/alice_wonderland_clean.txt data/alice_wonderland.txt
    echo "   Done! ($(wc -l < data/alice_wonderland.txt) lines)"
else
    echo "6. alice_wonderland.txt already exists"
fi
echo ""

# ============================================
# 7. WIKITEXT (Wikipedia articles)
# ============================================
if [ ! -f "data/wikitext.txt" ]; then
    echo "7. Downloading WikiText-2..."
    curl -L -o data/wikitext.txt \
        "https://cosmo.zip/pub/datasets/wikitext-2-raw/wiki.train.raw" \
        --silent --show-error 2>/dev/null
    if [ -s "data/wikitext.txt" ]; then
        echo "   Done! ($(wc -l < data/wikitext.txt) lines)"
    else
        echo "   Warning: Download failed, skipping..."
        rm -f data/wikitext.txt
    fi
else
    echo "7. wikitext.txt already exists"
fi
echo ""

# ============================================
# 8. CODE (from various sources)
# ============================================
if [ ! -f "data/code.txt" ]; then
    echo "7. Creating code dataset..."
    
    # Create a larger, diverse code dataset
    cat > data/code.txt << 'CODEEOF'
def hello_world():
    print("Hello, World!")
    return 0

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        return "Some sound"
    
    def eat(self, food):
        return f"Eating {food}"

class Dog(Animal):
    def speak(self):
        return "Woof!"
    
    def fetch(self, item):
        return f"Fetching {item}"

class Cat(Animal):
    def speak(self):
        return "Meow!"
    
    def purr(self):
        return "Purring..."

def count_words(text):
    words = text.split()
    return len(words)

def reverse_string(s):
    return s[::-1]

def is_palindrome(s):
    clean = ''.join(c.lower() for c in s if c.isalnum())
    return clean == clean[::-1]

def merge_dicts(dict1, dict2):
    result = dict1.copy()
    result.update(dict2)
    return result

def flatten_list(nested):
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result

def matrix_multiply(A, B):
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def sieve_of_eratosthenes(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def depth_first_search(graph, start):
    visited = set()
    stack = [start]
    result = []
    
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            result.append(vertex)
            stack.extend(neighbor for neighbor in graph[vertex] 
                       if neighbor not in visited)
    
    return result

def breadth_first_search(graph, start):
    visited = set()
    queue = [start]
    result = []
    
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            result.append(vertex)
            queue.extend(neighbor for neighbor in graph[vertex]
                       if neighbor not in visited)
    
    return result
CODEEOF
    echo "   Done! Created code samples"
else
    echo "7. code.txt already exists"
fi
echo ""

# ============================================
# 9. COMBINE ALL (except names and code)
# ============================================
if [ ! -f "data/training_data.txt" ]; then
    echo "9. Combining all books into training corpus..."
    {
        # Pride and Prejudice
        cat data/pride_prejudice.txt
        echo ""
        
        # Frankenstein  
        cat data/frankenstein.txt
        echo ""
        
        # Moby Dick
        cat data/moby_dick.txt
        echo ""
        
        # Sherlock Holmes
        cat data/sherlock_holmes.txt
        echo ""
        
        # Alice in Wonderland
        cat data/alice_wonderland.txt
        echo ""
        
        # WikiText
        if [ -f "data/wikitext.txt" ]; then
            cat data/wikitext.txt
        fi
    } > data/training_data.txt
    
    echo "   Done! ($(wc -l < data/training_data.txt) lines)"
else
    echo "9. training_data.txt already exists"
fi
echo ""

# ============================================
# SUMMARY
# ============================================
echo "=== Download & Clean Complete ==="
echo ""
echo "Individual files:"
ls -lh data/*.txt | grep -v training_data
echo ""
echo "Combined corpus:"
ls -lh data/training_data.txt
echo ""
echo "Usage:"
echo "  Names (for name generation):     data/names.txt"
echo "  Literature (for text):          data/training_data.txt"
echo "  Code (for code generation):      data/code.txt"
echo ""
echo "Training commands:"
echo "  Names:    ./microgpt --train --data data/names.txt --steps 50000"
echo "  Stories:  ./microgpt --train --data data/training_data.txt --steps 100000"
echo "  Code:     ./microgpt --train --data data/code.txt --steps 50000"

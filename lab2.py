import argparse
from collections import Counter
from urllib.parse import urlparse
import requests
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import sys
import unittest
from unittest.mock import patch, mock_open, MagicMock

def load_from_source(source):
   
    parsed = urlparse(source)
    if parsed.scheme in ("http", "https"):
        response = requests.get(source)
        response.raise_for_status()
        return response.content
    else:
        with open(source, "rb") as f:
            return f.read()


def build_unweighted_graph(nodes, adjacency_counts):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for (a, b), count in adjacency_counts.items():

        if a in nodes and b in nodes and not G.has_edge(a, b):
            G.add_edge(a, b)
    return G

def compute_distance_matrix(nodes, adjacency_counts, distance_mode="inverted"):
    n = len(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}  
    count_matrix = np.zeros((n, n), dtype=float)

    for (a, b), cnt in adjacency_counts.items():
        if a in node_index and b in node_index:
            i, j = node_index[a], node_index[b]
            count_matrix[i, j] = float(cnt)

    count_matrix = (count_matrix + count_matrix.T) / 2
    np.fill_diagonal(count_matrix, 0)

    if distance_mode == "direct":
        distance_matrix = count_matrix.copy()
    elif distance_mode == "inverted":
        max_count = np.max(count_matrix)
        if max_count > 0:
            distance_matrix = (max_count + 1) - count_matrix
        else:
            distance_matrix = np.ones_like(count_matrix, dtype=float)
    else:
        raise ValueError("distance_mode must be 'direct' or 'inverted'")

    np.fill_diagonal(distance_matrix, 0)
    return distance_matrix, count_matrix
def visualize_network(G, distance_matrix, nodes, node_colors=None, node_labels=None,
                       figsize=(14, 14), title="Network Graph", jitter=0.2, random_state=42):
    n = len(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}
    side = int(np.ceil(np.sqrt(n)))
    pos = {}
    rng = np.random.RandomState(random_state)
    for i, node in enumerate(nodes):
        row = i // side
        col = i % side
        jitter_x = rng.uniform(-jitter, jitter)
        jitter_y = rng.uniform(-jitter, jitter)
        pos[node] = (col + jitter_x, -row + jitter_y)
    
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_title(title)
    if node_colors is None:
        node_colors = ["lightblue"] * n
    if node_labels is None:
        node_labels = {node: str(node)[:10] for node in nodes}
    nx.draw_networkx_nodes(G, pos, node_size=900, node_color=node_colors,
                           edgecolors="black", linewidths=1, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.7, ax=ax)
    edge_labels = {}
    for u, v in G.edges():
        if u in node_index and v in node_index:
            dist_val = distance_matrix[node_index[u], node_index[v]]
            edge_labels[(u, v)] = f"{dist_val:.1f}"
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_size=8, ax=ax, label_pos=0.5)
    
    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.show()



def tokenize_text(text, char_like=["'"]):
    text = text.lower()
    tokens = []
    current_word = []
    punctuation = {',', '.'}
    for char in text:
        if char.isalpha() or char in char_like:
            current_word.append(char)
        elif char in punctuation:
            if current_word:
                tokens.append(''.join(current_word))
                current_word = []
            tokens.append(char)
        else:
            #if char.isdigit():
            #    continue
            if current_word:
                tokens.append(''.join(current_word))
                current_word = []
    if current_word:
        tokens.append(''.join(current_word))
        current_word = []
    # STEP 7: Return the list of tokens
    return tokens


def replace_rare_tokens(tokens, rare_threshold=0.01, rare_token="<RARE>"):
    
    punctuation = {',', '.'}

    word_tokens = [t for t in tokens if t not in punctuation] 
    total_words = len(word_tokens)
    if total_words == 0:
        return tokens[:], set(), Counter(tokens)

    word_counts = Counter(word_tokens) 
    rare_set = {word for word, count in word_counts.items() if count / total_words < rare_threshold}

    new_tokens = [rare_token if token in rare_set else token for token in tokens]
    
    final_counts = Counter(new_tokens)  
    return new_tokens, rare_set, final_counts

   

def get_text_adjacencies(tokens):
    adjacency_counts = Counter()

    for i in range(len(tokens) - 1): 
        current_token = tokens[i]  # Replace this line
        next_token = tokens[i+1]     
       
        if current_token != next_token:  # Replace 
            adjacency_counts[(current_token, next_token)] += 1  # Replace this line with your code

    # STEP 6: Return the Counter with all pair frequencies
    return adjacency_counts


def process_text_network(source, rare_threshold=0.01, rare_token="<RARE>",
                          distance_mode="inverted", verbose=False, nsample_tokens=20):
    # Load and tokenize
    content = load_from_source(source)
    text = content.decode('utf-8', errors='ignore')
    
    if verbose:
        print(f"Loaded text: {len(text)} characters")
    
    tokens = tokenize_text(text)
    
    if verbose:
        print(f"Tokenized: {len(tokens)} tokens")
        print(f"Sample tokens: {list(set(tokens))[:nsample_tokens]}")
    
    # Handle rare tokens
    processed_tokens, rare_set, token_counts = replace_rare_tokens(
        tokens, rare_threshold, rare_token)
    
    if verbose:
        print(f"Replaced {len(rare_set)} rare tokens (threshold={rare_threshold})")
        print(f"Final vocabulary: {len(token_counts)} unique tokens")
        print(f"Sample tokens: {list(set(processed_tokens))[:nsample_tokens]}")
    
    # Build adjacencies and graph
    adjacency_counts = get_text_adjacencies(processed_tokens)
    nodes = sorted(token_counts.keys(), key=lambda x: (-token_counts[x], x))
    graph = build_unweighted_graph(nodes, adjacency_counts)
    distance_matrix, count_matrix = compute_distance_matrix(nodes, adjacency_counts, distance_mode)
    
    if verbose:
        print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        print(f"Top tokens by frequency:")
        for i, node in enumerate(nodes[:10]):
            print(f"  {i+1:2d}. '{node}' (freq={token_counts[node]})")
    
    return {
        'graph': graph,
        'nodes': nodes,
        'adjacency_counts': adjacency_counts,
        'distance_matrix': distance_matrix,
        'count_matrix': count_matrix,
        'token_counts': token_counts,
        'rare_tokens': rare_set,
        'original_tokens': tokens
    }

def preprocess_image(image_data, target_size=(128, 128), quantize_levels=16):
    img = Image.open(BytesIO(image_data))
    
    # Handle different image modes
    if img.mode == 'RGBA':
        # Composite with white background
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    elif img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')
    
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(img)
    
    # Add channel dimension for grayscale
    if len(img_array.shape) == 2:
        img_array = img_array[:, :, np.newaxis]
    
    height, width, channels = img_array.shape
    quantized = np.zeros_like(img_array)
    quantization_info = {}
    
    # Quantize each channel independently using vectorized operations
    for c in range(channels):
        channel_data = img_array[:, :, c]
        min_val, max_val = channel_data.min(), channel_data.max()
        
        if min_val == max_val:
            # Uniform channel: all pixels map to level 0
            quantized[:, :, c] = 0
            quantization_info[c] = {'min': min_val, 'max': max_val, 'levels': [min_val]}
        else:
            # Create evenly-spaced quantization levels
            levels = np.linspace(min_val, max_val, quantize_levels)
            quantized[:, :, c] = np.searchsorted(levels, channel_data, side='left')
            # Clamp to valid range [0, quantize_levels-1]
            quantized[:, :, c] = np.clip(quantized[:, :, c], 0, quantize_levels - 1)
            
            quantization_info[c] = {'min': min_val, 'max': max_val, 'levels': levels}
    
    return quantized, quantization_info


def get_spatial_adjacencies(quantized_image):


    from collections import Counter

    height, width, channels = quantized_image.shape

    print(f"Image dimensions: {height}h × {width}w × {channels}c")  # Debug helper (optional)
    adjacency_counts = Counter()  # Replace this line
    color_frequencies = Counter()  # Replace this line
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  
    print(f"Using 4-connected neighbors: {directions}")  # Debug helper (optional)
    for i in range(height):  # Replace 0 with correct range
        for j in range(width):  
            current_color = tuple(quantized_image[i, j])  

            print(f"Pixel ({i},{j}): color={current_color}")  
            color_frequencies[current_color] +=1 
            for di, dj in directions:  
                ni =  i + di   # Replace this line
                nj = j + dj   # Replace this line

                # STEP 8: Check if neighbor is within image bounds
                # HINT: Check 0 <= ni < height and 0 <= nj < width
                if 0 <= ni < height and 0 <= nj < width:  # Replace False with correct condition

                    # STEP 9: Get neighbor's color as a tuple
                    # HINT: Same as step 5 but use neighbor coordinates (ni, nj)
                    neighbor_color = tuple(quantized_image[ni, nj])  # Replace this line

                    # STEP 10: Check if colors are different and record adjacency
                    # HINT: Only count if current_color != neighbor_color
                    if current_color != neighbor_color:  # Replace False with correct condition
                        adjacency_counts[(current_color, neighbor_color)] +=1


    print(f"Found {len(color_frequencies)} unique colors")  # Debug helper (optional)
    print(f"Found {len(adjacency_counts)} unique adjacencies")  # Debug helper (optional)
    unique_colors = sorted(color_frequencies.keys(), key=lambda c: (-color_frequencies[c], c))

    print(f"Top 3 colors: {unique_colors[:3]}")  # Debug helper (optional)

    # STEP 12: Return all three results as a tuple
    return adjacency_counts, unique_colors, color_frequencies


def color_to_rgb(color_tuple, quantization_info):
    rgb = []
    for c, quant_val in enumerate(color_tuple):
        if c < len(quantization_info):
            levels = quantization_info[c]['levels']
            if isinstance(levels, (list, np.ndarray)) and len(levels) > int(quant_val):
                rgb_val = levels[int(quant_val)]
            else:
                rgb_val = levels[0] if hasattr(levels, '__getitem__') else levels
            rgb.append(rgb_val / 255.0)  # Normalize to [0,1]
        else:
            rgb.append(0.5)
    
    # Handle different channel counts
    if len(rgb) == 1:
        return (rgb[0], rgb[0], rgb[0])  # Grayscale to RGB
    elif len(rgb) >= 3:
        return tuple(rgb[:3])
    else:
        return tuple((rgb + [0, 0, 0])[:3])


def show_quantized_image(quantized_image, quantization_info, figsize=(8, 8)):

    height, width, channels = quantized_image.shape
    display_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            color_tuple = tuple(quantized_image[i, j])
            rgb = color_to_rgb(color_tuple, quantization_info)
            display_image[i, j] = [int(c * 255) for c in rgb]
    
    plt.figure(figsize=figsize)
    plt.imshow(display_image)
    plt.title(f"Quantized Image ({height}x{width}, {channels} channels)")
    plt.axis('off')
    plt.show()


def process_image_network(source, target_size=(128, 128), quantize_levels=16,
                           distance_mode="inverted", verbose=True):
    content = load_from_source(source)
    quantized_image, quantization_info = preprocess_image(content, target_size, quantize_levels)
    
    if verbose:
        print(f"Image processed: {quantized_image.shape}, {quantize_levels} levels per channel")
    
    if verbose:
        # Show quantized image
        show_quantized_image(quantized_image, quantization_info)
    
    # Get spatial adjacencies
    adjacency_counts, unique_colors, color_frequencies = get_spatial_adjacencies(quantized_image)
    
    if verbose:
        print(f"Found {len(unique_colors)} unique colors")
        print(f"Total spatial adjacencies: {sum(adjacency_counts.values())}")
        print("Top colors by frequency:")
        for i, color in enumerate(unique_colors[:10]):
            print(f"  {i+1:2d}. {color} (freq={color_frequencies[color]})")
    
    # Build graph
    graph = build_unweighted_graph(unique_colors, adjacency_counts)
    distance_matrix, count_matrix = compute_distance_matrix(unique_colors, adjacency_counts, distance_mode)
    
    if verbose:
        print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    return {
        'graph': graph,
        'nodes': unique_colors,
        'adjacency_counts': adjacency_counts,
        'distance_matrix': distance_matrix,
        'count_matrix': count_matrix,
        'color_frequencies': color_frequencies,
        'quantized_image': quantized_image,
        'quantization_info': quantization_info
    }

class TestUnifiedNetworks(unittest.TestCase):

    def test_build_unweighted_graph(self):
    
        nodes = ['A', 'B', 'C']
        adjacency_counts = Counter({('A', 'B'): 3, ('B', 'C'): 2, ('A', 'C'): 1})

        G = build_unweighted_graph(nodes, adjacency_counts)

        self.assertIsInstance(G, nx.Graph)
        self.assertEqual(G.number_of_nodes(), 3)
        self.assertEqual(G.number_of_edges(), 3)
        self.assertTrue(G.has_edge('A', 'B'))
        self.assertTrue(G.has_edge('B', 'C'))
        self.assertTrue(G.has_edge('A', 'C'))
        self.assertEqual(set(G.nodes()), set(nodes))
    
    def test_compute_distance_matrix_inverted_mode(self):

        
        nodes = ['A', 'B']
        adjacency_counts = Counter({('A', 'B'): 4, ('B', 'A'): 2})

        distance_matrix, count_matrix = compute_distance_matrix(nodes, adjacency_counts, "inverted")

        expected_counts = np.array([[0.0, 3.0],
                                    [3.0, 0.0]])
        np.testing.assert_array_almost_equal(count_matrix, expected_counts, decimal=6)

        expected_dist = np.array([[0.0, 1.0],
                                  [1.0, 0.0]])
        np.testing.assert_array_almost_equal(distance_matrix, expected_dist, decimal=6)

        np.testing.assert_array_equal(np.diag(distance_matrix), np.array([0.0, 0.0]))
        np.testing.assert_array_equal(np.diag(count_matrix), np.array([0.0, 0.0]))

    
    def test_tokenize_text_basic(self):
        
        text = "Hello, world! How are you? 123abc"
        tokens = tokenize_text(text)
        # '!' and '?' ignored; numbers stripped; comma kept
        expected = ["hello", ",", "world", "how", "are", "you", "abc"]
        self.assertEqual(tokens, expected)

        # numbers inside a word removed
        text2 = "abc123def"
        tokens2 = tokenize_text(text2)
        self.assertEqual(tokens2, ["abcdef"])

    
    def test_get_spatial_adjacencies_basic(self):
        quantized_image = np.array([
            [[0,0,0], [1,1,1]],
            [[2,2,2], [3,3,3]]
        ], dtype=int)

        adjacency_counts, unique_colors, color_frequencies = get_spatial_adjacencies(quantized_image)

        self.assertEqual(len(unique_colors), 4)
        self.assertEqual(set(color_frequencies.values()), {1})

        # total directional adjacencies in 2x2 grid with 4-connectivity = 8
        self.assertEqual(sum(adjacency_counts.values()), 8)

        # check a couple of specific adjacencies exist (directional)
        self.assertGreater(adjacency_counts[((0,0,0), (1,1,1))], 0)
        self.assertGreater(adjacency_counts[((0,0,0), (2,2,2))], 0)


def run_tests():
    """Run all unit tests."""
    print("=" * 70)
    print("RUNNING UNIT TESTS")
    print("=" * 70)
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestUnifiedNetworks)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
        print(f"Total tests run: {result.testsRun}")
    else:
        print("❌ SOME TESTS FAILED!")
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Only run tests when script is executed directly
    success = run_tests()

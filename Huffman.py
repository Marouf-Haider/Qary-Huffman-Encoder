#!/usr/bin/env python3
"""
Q-ary Huffman Encoding Application

This is a standalone Python implementation of the Q-ary Huffman encoding algorithm
with visualization capabilities. It provides:
- Character frequency analysis
- Q-ary Huffman tree construction
- Encoding and decoding
- Compression metrics calculation
- Simple visualization
"""

import os
import math
import json
import time
import heapq
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union, Any
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==========================================================================
# Data Models
# ==========================================================================

class HuffmanNode:
    """Node in a Huffman tree"""
    def __init__(self, value: str, frequency: int, percentage: float):
        self.value = value  # Character value (empty for internal nodes)
        self.frequency = frequency  # Frequency count
        self.percentage = percentage  # Percentage of total
        self.children = []  # Child nodes
        self.code = ""  # Huffman code for this node
    
    def __lt__(self, other):
        # For heap comparison
        return self.frequency < other.frequency
    
    def to_dict(self) -> dict:
        """Convert node to dictionary for JSON serialization"""
        return {
            "value": self.value,
            "frequency": self.frequency,
            "percentage": self.percentage,
            "code": self.code,
            "children": [child.to_dict() for child in self.children]
        }

class FrequencyItem:
    """Represents frequency data for a symbol"""
    def __init__(self, symbol: str, frequency: int, percentage: float):
        self.symbol = symbol
        self.frequency = frequency
        self.percentage = percentage

class CompressionMetrics:
    """Stores metrics about the compression"""
    def __init__(self, original_size: int, encoded_size: int, space_savings: int,
                 compression_ratio: float, percent_reduction: float,
                 bits_per_symbol: float, entropy: float, encoding_efficiency: float):
        self.original_size = original_size
        self.encoded_size = encoded_size
        self.space_savings = space_savings
        self.compression_ratio = compression_ratio
        self.percent_reduction = percent_reduction
        self.bits_per_symbol = bits_per_symbol
        self.entropy = entropy
        self.encoding_efficiency = encoding_efficiency
    
    def to_dict(self) -> dict:
        """Convert metrics to dictionary for JSON serialization"""
        return {
            "originalSize": self.original_size,
            "encodedSize": self.encoded_size,
            "spaceSavings": self.space_savings,
            "compressionRatio": self.compression_ratio,
            "percentReduction": self.percent_reduction,
            "bitsPerSymbol": self.bits_per_symbol,
            "entropy": self.entropy,
            "encodingEfficiency": self.encoding_efficiency
        }

class QValuePerformance:
    """Performance data for a specific Q value"""
    def __init__(self, q_value: int, compression_ratio: float, 
                 bits_per_symbol: float, tree_depth: int, encoding_time: float):
        self.q_value = q_value
        self.compression_ratio = compression_ratio
        self.bits_per_symbol = bits_per_symbol
        self.tree_depth = tree_depth
        self.encoding_time = encoding_time
    
    def to_dict(self) -> dict:
        """Convert performance data to dictionary for JSON serialization"""
        return {
            "qValue": self.q_value,
            "compressionRatio": self.compression_ratio,
            "bitsPerSymbol": self.bits_per_symbol,
            "treeDepth": self.tree_depth,
            "encodingTime": self.encoding_time
        }

# ==========================================================================
# Huffman Algorithm Implementation
# ==========================================================================

def calculate_frequency(text: str) -> Dict[str, int]:
    """Calculate the frequency of each character in a string"""
    return dict(Counter(text))

def get_frequency_items(frequency_map: Dict[str, int], total_chars: int) -> List[FrequencyItem]:
    """Convert frequency map to frequency items with percentage"""
    return [
        FrequencyItem(symbol, frequency, (frequency / total_chars) * 100)
        for symbol, frequency in frequency_map.items()
    ]

def build_qary_huffman_tree(frequencies: Dict[str, int], q: int) -> HuffmanNode:
    """Build a Q-ary Huffman Tree"""
    # Calculate total frequency for percentage calculations
    total_frequency = sum(frequencies.values())
    
    # Create leaf nodes for each character
    nodes = [
        HuffmanNode(symbol, frequency, (frequency / total_frequency) * 100)
        for symbol, frequency in frequencies.items()
    ]
    
    # If we have fewer nodes than q, add dummy nodes to make it work
    while len(nodes) > 1 and (len(nodes) - 1) % (q - 1) != 0:
        nodes.append(HuffmanNode("", 0, 0))
    
    # Build priority queue
    heapq.heapify(nodes)
    
    # Build the tree by combining nodes until only one node remains
    while len(nodes) > 1:
        # Take the q least frequent nodes
        children_to_merge = []
        for _ in range(min(q, len(nodes))):
            children_to_merge.append(heapq.heappop(nodes))
        
        # Create a new internal node with these as children
        total_freq = sum(node.frequency for node in children_to_merge)
        total_percent = sum(node.percentage for node in children_to_merge)
        new_node = HuffmanNode("", total_freq, total_percent)
        new_node.children = children_to_merge
        
        # Add the new node back to the priority queue
        heapq.heappush(nodes, new_node)
    
    # Return the root node (should be only one left)
    return nodes[0] if nodes else HuffmanNode("", 0, 0)

def assign_huffman_codes(node: HuffmanNode, prefix: str = "", codes: Dict[str, str] = None) -> Dict[str, str]:
    """Assign codes to each node in the Huffman tree"""
    if codes is None:
        codes = {}
    
    if node.value:
        # This is a leaf node (character node)
        codes[node.value] = prefix
        node.code = prefix
    else:
        # Assign codes to children
        for i, child in enumerate(node.children):
            # For q-ary tree, the digit used is 0, 1, 2, ... (q-1)
            child_prefix = prefix + str(i)
            assign_huffman_codes(child, child_prefix, codes)
    
    return codes

def encode_text(text: str, codes: Dict[str, str]) -> str:
    """Encode text using Huffman codes"""
    encoded_text = ""
    
    for char in text:
        if char in codes:
            encoded_text += codes[char]
        else:
            raise ValueError(f"No code for character: {char}")
    
    return encoded_text

def decode_text(encoded_text: str, codes: Dict[str, str]) -> str:
    """Decode text using Huffman codes"""
    # Create a reverse lookup table for decoding
    reverse_mapping = {code: char for char, code in codes.items()}
    
    current = ""
    decoded = ""
    
    for bit in encoded_text:
        current += bit
        if current in reverse_mapping:
            decoded += reverse_mapping[current]
            current = ""
    
    if current:
        raise ValueError("Invalid encoded string")
    
    return decoded

def calculate_entropy(frequencies: Dict[str, int], total_chars: int) -> float:
    """Calculate entropy of the data"""
    entropy = 0
    
    for frequency in frequencies.values():
        probability = frequency / total_chars
        entropy -= probability * math.log2(probability)
    
    return entropy

def calculate_compression_metrics(original_text: str, encoded_text: str, 
                                 codes: Dict[str, str], frequencies: Dict[str, int]) -> CompressionMetrics:
    """Calculate compression metrics"""
    # Original size in bits (assuming 8 bits per ASCII character)
    original_size = len(original_text) * 8
    
    # Encoded size in bits
    encoded_size = len(encoded_text)
    
    # Space savings in bytes
    space_savings = (original_size - encoded_size) // 8
    
    # Compression ratio
    compression_ratio = original_size / max(encoded_size, 1)
    
    # Percent reduction
    percent_reduction = ((original_size - encoded_size) / original_size) * 100
    
    # Calculate average bits per symbol
    total_bits = 0
    total_frequency = 0
    
    for char, frequency in frequencies.items():
        if char in codes:
            total_bits += len(codes[char]) * frequency
            total_frequency += frequency
    
    bits_per_symbol = total_bits / max(total_frequency, 1)
    
    # Calculate entropy
    entropy = calculate_entropy(frequencies, len(original_text))
    
    # Calculate encoding efficiency (how close we are to the theoretical limit)
    encoding_efficiency = (entropy / max(bits_per_symbol, 0.001)) * 100
    
    return CompressionMetrics(
        original_size,
        encoded_size,
        space_savings,
        compression_ratio,
        percent_reduction,
        bits_per_symbol,
        entropy,
        encoding_efficiency
    )

def get_tree_depth(node: HuffmanNode) -> int:
    """Get the depth of the Huffman tree"""
    if not node.children or len(node.children) == 0:
        return 0
    
    return 1 + max(get_tree_depth(child) for child in node.children)

def get_q_value_performance(text: str, q_values: List[int]) -> List[QValuePerformance]:
    """Get performance metrics for different q values"""
    results = []
    
    for q in q_values:
        start_time = time.time()
        
        frequencies = calculate_frequency(text)
        tree = build_qary_huffman_tree(frequencies, q)
        codes = assign_huffman_codes(tree)
        encoded_text = encode_text(text, codes)
        metrics = calculate_compression_metrics(text, encoded_text, codes, frequencies)
        tree_depth = get_tree_depth(tree)
        
        end_time = time.time()
        encoding_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        results.append(QValuePerformance(
            q,
            metrics.compression_ratio,
            metrics.bits_per_symbol,
            tree_depth,
            encoding_time
        ))
    
    return results

# ==========================================================================
# Export and Utilities
# ==========================================================================

def export_encoded_text(encoded_text: str, filename: str = "encoded_output.txt") -> bool:
    """Export encoded text to a file"""
    try:
        with open(filename, 'w') as f:
            f.write(encoded_text)
        return True
    except Exception as e:
        print(f"Error exporting encoded text: {e}")
        return False

def export_standalone_decoder(huffman_codes: Dict[str, str], filename: str = "huffman_decoder.py") -> bool:
    """Export a standalone Python decoder module"""
    try:
        decoder_code = f"""#!/usr/bin/env python3
\"\"\"
Standalone Huffman Decoder
Generated by Q-ary Huffman Encoder
\"\"\"

# Huffman codes for decoding
HUFFMAN_CODES = {huffman_codes}

def decode(encoded_text):
    \"\"\"Decode text that was encoded with Huffman coding\"\"\"
    # Create a reverse lookup table for decoding
    reverse_mapping = {{code: char for char, code in HUFFMAN_CODES.items()}}
    
    current = ""
    decoded = ""
    
    for bit in encoded_text:
        current += bit
        if current in reverse_mapping:
            decoded += reverse_mapping[current]
            current = ""
    
    if current:
        raise ValueError("Invalid encoded string")
    
    return decoded

# Example usage
if __name__ == "__main__":
    # Example encoded text (replace with your encoded text)
    example_encoded = "0110101001"
    
    try:
        result = decode(example_encoded)
        print(f"Decoded result: {{result}}")
    except ValueError as e:
        print(f"Error: {{e}}")
"""
        
        with open(filename, 'w') as f:
            f.write(decoder_code)
        return True
    except Exception as e:
        print(f"Error exporting decoder: {e}")
        return False

def export_full_package(original_text: str, encoded_text: str, huffman_codes: Dict[str, str], 
                        tree: HuffmanNode, metrics: CompressionMetrics, 
                        q_value: int, filename: str = "huffman_package") -> bool:
    """Export a full package with all encoding data and tools"""
    try:
        # Create a directory for the package if it doesn't exist
        os.makedirs(filename, exist_ok=True)
        
        # Export the encoded text
        with open(f"{filename}/encoded_output.txt", 'w') as f:
            f.write(encoded_text)
        
        # Export the decoder
        export_standalone_decoder(huffman_codes, f"{filename}/huffman_decoder.py")
        
        # Export the encoding data as JSON
        encoding_data = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "qValue": q_value,
                "originalLength": len(original_text)
            },
            "huffmanCodes": huffman_codes,
            "compressionMetrics": metrics.to_dict()
        }
        
        with open(f"{filename}/encoding_data.json", 'w') as f:
            json.dump(encoding_data, f, indent=2)
        
        # Create a README file
        readme_content = f"""# Huffman Encoding Package - by Haider Marouf -
Generated by Q-ary Huffman Encoder on {time.strftime("%Y-%m-%d %H:%M:%S")}

## Contents
- encoded_output.txt - The encoded output as a binary string
- huffman_decoder.py - A standalone Python decoder module
- encoding_data.json - Encoding data including Huffman codes and metrics

## Compression Summary
- Q Value: {q_value}
- Compression Ratio: {metrics.compression_ratio:.2f}:1
- Original Size: {metrics.original_size} bits ({len(original_text)} characters)
- Encoded Size: {metrics.encoded_size} bits
- Space Savings: {metrics.space_savings} bytes
- Bits Per Symbol: {metrics.bits_per_symbol:.2f}
- Encoding Efficiency: {metrics.encoding_efficiency:.1f}%

## Usage
To decode the encoded text:

```python
from huffman_decoder import decode

with open('encoded_output.txt', 'r') as f:
    encoded_text = f.read()

decoded_text = decode(encoded_text)
print(decoded_text)
```
"""
        
        with open(f"{filename}/README.md", 'w') as f:
            f.write(readme_content)
            
        return True
    except Exception as e:
        print(f"Error exporting package: {e}")
        return False

def format_bytes(bytes_count, decimals=2):
    """Format bytes to human-readable format"""
    if bytes_count == 0:
        return "0 Bytes"
    
    k = 1024
    sizes = ["Bytes", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
    i = int(math.floor(math.log(bytes_count, k)))
    
    return f"{bytes_count / math.pow(k, i):.{decimals}f} {sizes[i]}"

# ==========================================================================
# Visualization Functions
# ==========================================================================

def plot_frequency_histogram(frequency_items: List[FrequencyItem], ax=None):
    """Create a histogram visualization of character frequencies"""
    # Sort by frequency in descending order and take top 15 for visibility
    items = sorted(frequency_items, key=lambda x: x.frequency, reverse=True)[:15]
    
    # Extract data for plotting
    symbols = [item.symbol if item.symbol != ' ' else '␣' for item in items]
    frequencies = [item.frequency for item in items]
    
    # Create colors
    colors = ['#1976d2' if i % 2 == 0 else '#42a5f5' for i in range(len(symbols))]
    
    # Create a new figure if ax is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    
    # Create the bar chart
    bars = ax.bar(symbols, frequencies, color=colors)
    
    # Add labels and title
    ax.set_xlabel('Symbol')
    ax.set_ylabel('Frequency')
    ax.set_title('Top Symbol Frequencies')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}',
                ha='center', va='bottom', rotation=0)
    
    # Adjust layout
    plt.tight_layout()
    
    return ax

def plot_qvalue_comparison(performance_data: List[QValuePerformance], current_q: int, ax=None):
    """Create a comparison chart for different Q values"""
    # Extract data for plotting
    q_values = [str(perf.q_value) for perf in performance_data]
    compression_ratios = [perf.compression_ratio for perf in performance_data]
    bits_per_symbol = [perf.bits_per_symbol for perf in performance_data]
    
    # Highlight current Q value
    colors = ['#0d47a1' if perf.q_value == current_q else '#1976d2' for perf in performance_data]
    
    # Create a new figure if ax is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    
    # Create the bar chart for compression ratios
    ax.bar(q_values, compression_ratios, color=colors, alpha=0.7, label='Compression Ratio')
    
    # Add a second y-axis for bits per symbol
    ax2 = ax.twinx()
    ax2.plot(q_values, bits_per_symbol, 'o-', color='#ff9800', linewidth=2, markersize=8, label='Bits Per Symbol')
    
    # Add labels and title
    ax.set_xlabel('Q Value')
    ax.set_ylabel('Compression Ratio')
    ax2.set_ylabel('Bits Per Symbol')
    ax.set_title('Performance Comparison by Q Value')
    
    # Add legends
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    return ax

def create_tree_visualization(node: HuffmanNode, canvas_width=800, canvas_height=600):
    """Create a simple visualization of the Huffman tree"""
    import tkinter as tk
    from tkinter import Canvas
    
    root = tk.Toplevel()
    root.title("Huffman Tree Visualization")
    
    canvas = Canvas(root, width=canvas_width, height=canvas_height, bg='white')
    canvas.pack(fill=tk.BOTH, expand=True)
    
    # Add scrollbars for large trees
    h_scrollbar = tk.Scrollbar(root, orient=tk.HORIZONTAL, command=canvas.xview)
    v_scrollbar = tk.Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview)
    
    canvas.config(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
    
    h_scrollbar.pack(fill=tk.X, side=tk.BOTTOM)
    v_scrollbar.pack(fill=tk.Y, side=tk.RIGHT)
    
    # Make the canvas scrollable
    canvas.configure(scrollregion=(0, 0, 2000, 1000))
    
    # Calculate tree metrics
    depth = get_tree_depth(node)
    
    # Draw the tree
    def draw_node(node, x, y, level=0, max_width=1000, parent_x=None, parent_y=None):
        # Draw connection line to parent if not root
        if parent_x is not None and parent_y is not None:
            canvas.create_line(parent_x, parent_y, x, y, fill='#616e7c', width=1.5)
        
        # Draw node circle
        radius = 20
        if node.value:  # Leaf node
            fill_color = '#ff9800'
            stroke_color = '#f57c00'
        else:  # Internal node
            fill_color = '#1976d2'
            stroke_color = '#0d47a1'
        
        canvas.create_oval(x-radius, y-radius, x+radius, y+radius, fill=fill_color, outline=stroke_color, width=2)
        
        # Draw text inside circle
        if node.value:
            display_value = "␣" if node.value == " " else node.value
            canvas.create_text(x, y, text=display_value, fill='white', font=('Arial', 10, 'bold'))
        else:
            canvas.create_text(x, y, text=f"{node.frequency}", fill='white', font=('Arial', 10))
        
        # Draw frequency percentage below
        canvas.create_text(x, y+radius+15, text=f"{node.percentage:.1f}%", fill='#616e7c', font=('Arial', 8))
        
        # Draw code below if available
        if node.code:
            canvas.create_text(x, y+radius+30, text=node.code, fill='#323f4b', font=('Arial', 9, 'bold'))
        
        # Calculate spacing for children
        if node.children:
            # We distribute children width evenly
            child_width = max_width / len(node.children)
            next_level_y = y + 80  # Vertical distance between levels
            
            for i, child in enumerate(node.children):
                # Calculate x position for this child
                child_x = x - (max_width / 2) + (i * child_width) + (child_width / 2)
                
                # Draw edge label (the code digit)
                edge_label_x = (x + child_x) / 2
                edge_label_y = (y + next_level_y) / 2
                canvas.create_text(edge_label_x, edge_label_y, text=str(i), fill='#1976d2', font=('Arial', 10, 'bold'))
                
                # Recursively draw child
                draw_node(child, child_x, next_level_y, level+1, child_width, x, y)
    
    # Start drawing from root at center top
    draw_node(node, canvas_width//2, 50, max_width=canvas_width*0.8)
    
    # Add a legend
    legend_y = canvas_height - 40
    # Internal node
    canvas.create_oval(20, legend_y-10, 40, legend_y+10, fill='#1976d2', outline='#0d47a1', width=2)
    canvas.create_text(80, legend_y, text="Internal Node", anchor='w', fill='#323f4b')
    # Leaf node
    canvas.create_oval(200, legend_y-10, 220, legend_y+10, fill='#ff9800', outline='#f57c00', width=2)
    canvas.create_text(260, legend_y, text="Symbol Node", anchor='w', fill='#323f4b')
    
    return root

# ==========================================================================
# GUI Application
# ==========================================================================

class HuffmanEncoderApp:
    """Tkinter GUI application for Huffman encoding"""
    
    def __init__(self, master):
        self.master = master
        master.title("Q-ary Huffman Encoder - by Haider Marouf")
        master.geometry("1200x600")
        footer = ttk.Label(master, text="Made by Marouf Haider - National Higher School of Mathematics -", font=("Arial", 9), anchor="e")
        footer.pack(side=tk.BOTTOM, pady=2)

        # Initialize data structures
        self.input_text = ""
        self.q_value = 2
        self.frequencies = {}
        self.frequency_items = []
        self.huffman_tree = None
        self.huffman_codes = {}
        self.encoded_text = ""
        self.compression_metrics = None
        self.q_value_performance = []
        
        # Create the main notebook with tabs
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        # Create tabs
        self.tab_input = ttk.Frame(self.notebook)
        self.tab_frequency = ttk.Frame(self.notebook)
        self.tab_tree = ttk.Frame(self.notebook)
        self.tab_encoded = ttk.Frame(self.notebook)
        self.tab_metrics = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_input, text="Input")
        self.notebook.add(self.tab_frequency, text="Frequency Analysis")
        self.notebook.add(self.tab_tree, text="Huffman Tree")
        self.notebook.add(self.tab_encoded, text="Encoded Output")
        self.notebook.add(self.tab_metrics, text="Compression Metrics")
        
        # Set up each tab
        self.setup_input_tab()
        self.setup_frequency_tab()
        self.setup_tree_tab()
        self.setup_encoded_tab()
        self.setup_metrics_tab()
        
        # Status bar
        self.status_bar = tk.Label(master, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def setup_input_tab(self):
        """Set up the input tab with text area and controls"""
        # Main frame for input
        input_frame = ttk.Frame(self.tab_input, padding=10)
        input_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(input_frame, text="Q-ary Huffman Encoder", font=("Arial", 16, "bold")).pack(pady=(0, 10))
        
        # Input method selection
        method_frame = ttk.Frame(input_frame)
        method_frame.pack(fill=tk.X, pady=5)
        
        self.input_method = tk.StringVar(value="text")
        ttk.Radiobutton(method_frame, text="Text Input", variable=self.input_method, 
                        value="text", command=self.toggle_input_method).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(method_frame, text="File Upload", variable=self.input_method, 
                        value="file", command=self.toggle_input_method).pack(side=tk.LEFT, padx=5)
        
        # Text input frame
        self.text_input_frame = ttk.Frame(input_frame)
        self.text_input_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        ttk.Label(self.text_input_frame, text="Enter text to encode:").pack(anchor=tk.W)
        
        self.text_input = scrolledtext.ScrolledText(self.text_input_frame, wrap=tk.WORD, height=12)
        self.text_input.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # File input frame (initially hidden)
        self.file_input_frame = ttk.Frame(input_frame)
        
        ttk.Label(self.file_input_frame, text="Upload a file to encode:").pack(anchor=tk.W)
        
        file_button_frame = ttk.Frame(self.file_input_frame)
        file_button_frame.pack(fill=tk.X, pady=5)
        
        self.file_path_var = tk.StringVar()
        ttk.Entry(file_button_frame, textvariable=self.file_path_var, state='readonly', width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_button_frame, text="Browse...", command=self.browse_file).pack(side=tk.LEFT)
        
        self.file_info_label = ttk.Label(self.file_input_frame, text="No file selected")
        self.file_info_label.pack(anchor=tk.W, pady=5)
        
        # Q value selection
        q_frame = ttk.Frame(input_frame)
        q_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(q_frame, text="Q Value (encoding base):").pack(side=tk.LEFT, padx=5)
        
        self.q_var = tk.IntVar(value=2)
        q_values = [2, 3, 4, 5, 8]
        q_combo = ttk.Combobox(q_frame, textvariable=self.q_var, values=q_values, width=5, state="readonly")
        q_combo.pack(side=tk.LEFT, padx=5)
        
        info_text = ("The Q value determines how many branches each node in the Huffman tree can have. "
                     "A higher Q value can sometimes achieve better compression for certain types of data.")
        ttk.Label(q_frame, text=info_text, wraplength=600, font=("Arial", 9, "italic")).pack(side=tk.LEFT, padx=10)
        
        # Action buttons
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.encode_button = ttk.Button(button_frame, text="Encode", command=self.encode_data)
        self.encode_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Clear", command=self.clear_input).pack(side=tk.LEFT, padx=5)
    
    def setup_frequency_tab(self):
        """Set up the frequency analysis tab"""
        # Main frame for frequency
        freq_frame = ttk.Frame(self.tab_frequency, padding=10)
        freq_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(freq_frame, text="Frequency Analysis", font=("Arial", 14, "bold")).pack(pady=(0, 10))
        
        # Split into table and chart panels
        panel_frame = ttk.Frame(freq_frame)
        panel_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frequency table frame (left side)
        table_frame = ttk.Frame(panel_frame)
        table_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        ttk.Label(table_frame, text="Symbol Frequencies", font=("Arial", 12)).pack(anchor=tk.W, pady=(0, 5))
        
        # Create table
        table_columns = ('symbol', 'frequency', 'percentage', 'code')
        self.freq_table = ttk.Treeview(table_frame, columns=table_columns, show='headings')
        
        # Define headings
        self.freq_table.heading('symbol', text='Symbol')
        self.freq_table.heading('frequency', text='Frequency')
        self.freq_table.heading('percentage', text='Percentage')
        self.freq_table.heading('code', text='Code')
        
        # Define columns
        self.freq_table.column('symbol', width=80, anchor=tk.CENTER)
        self.freq_table.column('frequency', width=80, anchor=tk.CENTER)
        self.freq_table.column('percentage', width=80, anchor=tk.CENTER)
        self.freq_table.column('code', width=150, anchor=tk.W)
        
        # Add a scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.freq_table.yview)
        self.freq_table.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.freq_table.pack(fill=tk.BOTH, expand=True)
        
        # Sorting callback
        def sort_treeview(col, reverse):
            l = [(self.freq_table.set(k, col), k) for k in self.freq_table.get_children('')]
            try:
                # Try to convert to numeric for proper sorting
                if col in ('frequency', 'percentage'):
                    l = [(float(val), k) for val, k in l]
            except ValueError:
                pass
            
            l.sort(reverse=reverse)
            for index, (_, k) in enumerate(l):
                self.freq_table.move(k, '', index)
            
            # Reverse sort next time
            self.freq_table.heading(col, command=lambda: sort_treeview(col, not reverse))
        
        # Configure column headings for sorting
        for col in table_columns:
            self.freq_table.heading(col, command=lambda _col=col: sort_treeview(_col, False))
        
        # Chart frame (right side)
        chart_frame = ttk.Frame(panel_frame)
        chart_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        ttk.Label(chart_frame, text="Frequency Distribution", font=("Arial", 12)).pack(anchor=tk.W, pady=(0, 5))
        
        # Create figure for matplotlib chart
        self.freq_figure = plt.Figure(figsize=(5, 4), dpi=100)
        self.freq_chart = FigureCanvasTkAgg(self.freq_figure, chart_frame)
        self.freq_chart.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Export button for frequency data
        ttk.Button(freq_frame, text="Export Frequency Data", command=self.export_frequency_data).pack(anchor=tk.E, pady=5)
    
    def setup_tree_tab(self):
        """Set up the Huffman tree visualization tab"""
        # Main frame for tree
        tree_frame = ttk.Frame(self.tab_tree, padding=10)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(tree_frame, text="Huffman Tree Visualization", font=("Arial", 14, "bold")).pack(pady=(0, 10))
        
        # Tree visualization controls
        controls_frame = ttk.Frame(tree_frame)
        controls_frame.pack(fill=tk.X, pady=5)
        
        # Checkboxes for display options
        self.show_frequencies = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls_frame, text="Show Frequencies", variable=self.show_frequencies).pack(side=tk.LEFT, padx=5)
        
        self.show_codes = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls_frame, text="Show Codes", variable=self.show_codes).pack(side=tk.LEFT, padx=5)
        
        # Placeholder for tree visualization
        self.tree_placeholder = ttk.Label(tree_frame, text="Encode data to generate Huffman tree visualization",
                                           font=("Arial", 12), foreground="gray")
        self.tree_placeholder.pack(fill=tk.BOTH, expand=True, pady=50)
        
        # Button to open interactive tree view
        self.tree_button = ttk.Button(tree_frame, text="Open Tree Visualization", command=self.show_tree_visualization)
        self.tree_button.pack(pady=10)
        self.tree_button.configure(state=tk.DISABLED)
        
        # Tree info
        info_frame = ttk.Frame(tree_frame)
        info_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(info_frame, text="Tree Information:", font=("Arial", 12)).pack(anchor=tk.W)
        
        self.tree_info_text = tk.Text(info_frame, height=5, wrap=tk.WORD)
        self.tree_info_text.pack(fill=tk.X, pady=5)
        self.tree_info_text.insert(tk.END, "No tree data available yet. Encode some text to generate a Huffman tree.")
        self.tree_info_text.config(state=tk.DISABLED)
    
    def setup_encoded_tab(self):
        """Set up the encoded output tab"""
        # Main frame for encoded output
        encoded_frame = ttk.Frame(self.tab_encoded, padding=10)
        encoded_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(encoded_frame, text="Encoded Output", font=("Arial", 14, "bold")).pack(pady=(0, 10))
        
        # Encoded text display
        ttk.Label(encoded_frame, text="Encoded Text:", font=("Arial", 12)).pack(anchor=tk.W)
        
        self.encoded_text_box = scrolledtext.ScrolledText(encoded_frame, wrap=tk.WORD, height=10, font=("Courier", 10))
        self.encoded_text_box.pack(fill=tk.BOTH, expand=True, pady=5)
        self.encoded_text_box.insert(tk.END, "Encode data to generate encoded output.")
        self.encoded_text_box.config(state=tk.DISABLED)
        
        # Action buttons for encoded text
        button_frame = ttk.Frame(encoded_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="Copy to Clipboard", command=self.copy_encoded_text).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Encoded Text", command=self.save_encoded_text).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Decoder Module", command=self.save_decoder_module).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Export Full Package", command=self.export_package).pack(side=tk.LEFT, padx=5)
        
        # Decoding test section
        decode_frame = ttk.LabelFrame(encoded_frame, text="Decoding Test", padding=10)
        decode_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(decode_frame, text="Enter encoded text to decode:").pack(anchor=tk.W)
        
        self.decode_input = scrolledtext.ScrolledText(decode_frame, height=4, font=("Courier", 10))
        self.decode_input.pack(fill=tk.X, pady=5)
        
        ttk.Button(decode_frame, text="Decode", command=self.test_decode).pack(anchor=tk.W)
        
        ttk.Label(decode_frame, text="Decoded result:").pack(anchor=tk.W, pady=(10, 0))
        
        self.decode_result = scrolledtext.ScrolledText(decode_frame, height=4, font=("Courier", 10))
        self.decode_result.pack(fill=tk.X, pady=5)
        self.decode_result.config(state=tk.DISABLED)
    
    def setup_metrics_tab(self):
        """Set up the compression metrics tab"""
        # Main frame for metrics
        metrics_frame = ttk.Frame(self.tab_metrics, padding=10)
        metrics_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(metrics_frame, text="Compression Metrics", font=("Arial", 14, "bold")).pack(pady=(0, 10))
        
        # Key metrics grid
        # === Row 1: Top Section (2 columns) ===
        top_section = ttk.Frame(metrics_frame)
        top_section.pack(fill=tk.BOTH, expand=True)

        # Left Column: Metric Cards (2x2 grid)
        metrics_grid = ttk.Frame(top_section)
        metrics_grid.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        metric_cards = [
            {"name": "Compression Ratio", "id": "ratio"},
            {"name": "Space Savings", "id": "savings"},
            {"name": "Bits Per Symbol", "id": "bits"},
            {"name": "Entropy", "id": "entropy"}
        ]

        for i, metric in enumerate(metric_cards):
            card = ttk.LabelFrame(metrics_grid, text=metric["name"], padding=10)
            card.grid(row=i//2, column=i%2, padx=5, pady=5, sticky=tk.NSEW)

            value_var = tk.StringVar(value="N/A")
            setattr(self, f"metric_{metric['id']}_var", value_var)
            ttk.Label(card, textvariable=value_var, font=("Arial", 14, "bold")).pack(pady=5)

            note_var = tk.StringVar(value="")
            setattr(self, f"metric_{metric['id']}_note_var", note_var)
            ttk.Label(card, textvariable=note_var, font=("Arial", 9)).pack()

        metrics_grid.columnconfigure(0, weight=1)
        metrics_grid.columnconfigure(1, weight=1)

        # Right Column: Q-Value Performance Table
        perf_frame = ttk.Frame(top_section)
        perf_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(perf_frame, text="Q-Value Performance", font=("Arial", 12, "bold")).pack(anchor=tk.W)

        columns = ('q_value', 'ratio', 'bits', 'depth', 'time')
        self.perf_table = ttk.Treeview(perf_frame, columns=columns, show='headings', height=5)

        for col, label in zip(columns, ['Q Value', 'Compression Ratio', 'Bits Per Symbol', 'Tree Depth', 'Encoding Time (ms)']):
            self.perf_table.heading(col, text=label)
            self.perf_table.column(col, anchor=tk.CENTER)

        scrollbar = ttk.Scrollbar(perf_frame, orient=tk.VERTICAL, command=self.perf_table.yview)
        self.perf_table.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.perf_table.pack(fill=tk.BOTH, expand=True)

        # === Row 2: Bottom Section (Chart) ===
        ttk.Label(metrics_frame, text="Performance Visualization", font=("Arial", 12)).pack(anchor=tk.W, pady=(20, 5))

        chart_wrapper = ttk.Frame(metrics_frame, height=300)
        chart_wrapper.pack(fill=tk.X, pady=(0, 10))
        chart_wrapper.pack_propagate(False)

        self.perf_figure = plt.Figure(figsize=(8, 3.5), dpi=100)
        self.perf_chart = FigureCanvasTkAgg(self.perf_figure, chart_wrapper)
        self.perf_chart.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # =============== Event Handlers and Core Functionality ===============
    
    def toggle_input_method(self):
        """Toggle between text input and file upload methods"""
        method = self.input_method.get()
        
        if method == "text":
            self.file_input_frame.pack_forget()
            self.text_input_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        else:  # file
            self.text_input_frame.pack_forget()
            self.file_input_frame.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def browse_file(self):
        """Open file dialog to select a file to encode"""
        file_path = filedialog.askopenfilename(
            title="Select a file to encode",
            filetypes=[
                ("Text files", "*.txt"),
                ("CSV files", "*.csv"),
                ("HTML files", "*.html"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.file_path_var.set(file_path)
            
            # Get file info
            file_size = os.path.getsize(file_path)
            file_name = os.path.basename(file_path)
            
            # Update file info label
            self.file_info_label.config(text=f"File: {file_name} ({format_bytes(file_size)})")
    
    def clear_input(self):
        """Clear the input fields"""
        self.text_input.delete(1.0, tk.END)
        self.file_path_var.set("")
        self.file_info_label.config(text="No file selected")
    
    def encode_data(self):
        """Encode the input text with Huffman coding"""
        # Get the input method
        method = self.input_method.get()
        
        # Get input text based on the method
        if method == "text":
            input_text = self.text_input.get(1.0, tk.END).strip()
            if not input_text:
                messagebox.showerror("Error", "Please enter some text to encode.")
                return
            self.input_text = input_text
        else:  # file
            file_path = self.file_path_var.get()
            if not file_path:
                messagebox.showerror("Error", "Please select a file to encode.")
                return
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.input_text = f.read()
            except Exception as e:
                messagebox.showerror("Error", f"Could not read the file: {str(e)}")
                return
        
        # Get the Q value
        q_value = self.q_var.get()
        self.q_value = q_value
        
        # Show encoding status
        self.status_bar.config(text="Encoding data...")
        self.encode_button.config(state=tk.DISABLED)
        self.master.update()
        
        try:
            # Start timing
            start_time = time.time()
            
            # Calculate character frequencies
            self.frequencies = calculate_frequency(self.input_text)
            self.frequency_items = get_frequency_items(self.frequencies, len(self.input_text))
            
            # Build Huffman tree
            self.huffman_tree = build_qary_huffman_tree(self.frequencies, q_value)
            
            # Assign codes to symbols
            self.huffman_codes = assign_huffman_codes(self.huffman_tree)
            
            # Encode the text
            self.encoded_text = encode_text(self.input_text, self.huffman_codes)
            
            # Calculate compression metrics
            self.compression_metrics = calculate_compression_metrics(
                self.input_text, self.encoded_text, self.huffman_codes, self.frequencies
            )
            
            # Get performance data for different q values
            q_values = [2, 3, 4, 5, 8]
            self.q_value_performance = get_q_value_performance(self.input_text, q_values)
            
            # End timing
            end_time = time.time()
            encoding_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            # Update the UI with the results
            self.update_frequency_tab()
            self.update_tree_tab()
            self.update_encoded_tab()
            self.update_metrics_tab()
            
            # Switch to the frequency analysis tab
            self.notebook.select(self.tab_frequency)
            
            # Show success message
            self.status_bar.config(text=f"Encoding completed in {encoding_time:.1f} ms. Compression ratio: {self.compression_metrics.compression_ratio:.2f}:1")
            messagebox.showinfo("Success", f"Encoding completed successfully!\nAchieved {self.compression_metrics.compression_ratio:.2f}:1 compression ratio.")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during encoding: {str(e)}")
            self.status_bar.config(text="Error during encoding")
        
        finally:
            self.encode_button.config(state=tk.NORMAL)
    
    def update_frequency_tab(self):
        """Update the frequency analysis tab with current data"""
        # Clear existing entries in the table
        for i in self.freq_table.get_children():
            self.freq_table.delete(i)
        
        # Add new frequency data to the table
        for item in self.frequency_items:
            display_symbol = "␣" if item.symbol == " " else item.symbol
            self.freq_table.insert("", tk.END, values=(
                display_symbol,
                item.frequency,
                f"{item.percentage:.1f}%",
                self.huffman_codes.get(item.symbol, "")
            ))
        
        # Update the chart
        self.freq_figure.clear()
        ax = self.freq_figure.add_subplot(111)
        plot_frequency_histogram(self.frequency_items, ax)
        self.freq_chart.draw()
    
    def update_tree_tab(self):
        """Update the Huffman tree tab with current data"""
        # Enable the tree visualization button
        self.tree_button.configure(state=tk.NORMAL)
        
        # Update the placeholder text
        self.tree_placeholder.config(text="Click 'Open Tree Visualization' to view the Huffman tree")
        
        # Update tree info text
        tree_depth = get_tree_depth(self.huffman_tree)
        num_internal_nodes = sum(1 for _ in self.traverse_internal_nodes(self.huffman_tree))
        num_leaf_nodes = len(self.huffman_codes)
        
        tree_info = (f"Tree Information:\n"
                     f"- Q value: {self.q_value}\n"
                     f"- Tree depth: {tree_depth}\n"
                     f"- Number of internal nodes: {num_internal_nodes}\n"
                     f"- Number of leaf nodes (symbols): {num_leaf_nodes}\n"
                     f"- Total nodes: {num_internal_nodes + num_leaf_nodes}")
        
        self.tree_info_text.config(state=tk.NORMAL)
        self.tree_info_text.delete(1.0, tk.END)
        self.tree_info_text.insert(tk.END, tree_info)
        self.tree_info_text.config(state=tk.DISABLED)
    
    def traverse_internal_nodes(self, node):
        """Generator to traverse internal nodes of the tree"""
        if not node.value:  # This is an internal node
            yield node
            for child in node.children:
                yield from self.traverse_internal_nodes(child)
    
    def update_encoded_tab(self):
        """Update the encoded output tab with current data"""
        # Update encoded text display
        self.encoded_text_box.config(state=tk.NORMAL)
        self.encoded_text_box.delete(1.0, tk.END)
        self.encoded_text_box.insert(tk.END, self.encoded_text)
        self.encoded_text_box.config(state=tk.DISABLED)
        
        # Enable the decode input by setting example
        self.decode_input.delete(1.0, tk.END)
        # Take a small sample of the encoded text as an example
        sample_length = min(20, len(self.encoded_text))
        self.decode_input.insert(tk.END, self.encoded_text[:sample_length])
    
    def update_metrics_tab(self):
        """Update the compression metrics tab with current data"""
        metrics = self.compression_metrics
        
        # Update metric cards
        self.metric_ratio_var.set(f"{metrics.compression_ratio:.2f}:1")
        self.metric_ratio_note_var.set(f"Original size reduced by {metrics.percent_reduction:.1f}%")
        
        self.metric_savings_var.set(f"{metrics.space_savings} bytes")
        self.metric_savings_note_var.set(f"Original: {metrics.original_size} bits, Encoded: {metrics.encoded_size} bits")
        
        self.metric_bits_var.set(f"{metrics.bits_per_symbol:.2f} bits")
        self.metric_bits_note_var.set(f"Compared to 8 bits in standard ASCII")
        
        self.metric_entropy_var.set(f"{metrics.entropy:.2f} bits")
        self.metric_entropy_note_var.set(f"Encoding efficiency: {metrics.encoding_efficiency:.1f}%")
        
        # Clear existing entries in the performance table
        for i in self.perf_table.get_children():
            self.perf_table.delete(i)
        
        # Add new performance data to the table
        for perf in self.q_value_performance:
            q_name = f"{perf.q_value}"
            if perf.q_value == 2:
                q_name += " (Binary)"
            elif perf.q_value == 3:
                q_name += " (Ternary)"
            elif perf.q_value == 4:
                q_name += " (Quaternary)"
            elif perf.q_value == 5:
                q_name += " (Quinary)"
            elif perf.q_value == 8:
                q_name += " (Octal)"
            
            # Highlight current Q value
            tags = ('current',) if perf.q_value == self.q_value else ()
            
            self.perf_table.insert("", tk.END, values=(
                q_name,
                f"{perf.compression_ratio:.2f}:1",
                f"{perf.bits_per_symbol:.2f} bits",
                f"{perf.tree_depth} levels",
                f"{perf.encoding_time:.1f} ms"
            ), tags=tags)
        
        # Configure tag for highlighting
        self.perf_table.tag_configure('current', background='#e3f2fd')
        
        # Update the chart
        self.perf_figure.clear()
        ax = self.perf_figure.add_subplot(111)
        plot_qvalue_comparison(self.q_value_performance, self.q_value, ax)
        self.perf_chart.draw()
    
    def show_tree_visualization(self):
        """Open a new window with interactive tree visualization"""
        if self.huffman_tree:
            create_tree_visualization(self.huffman_tree)
    
    def copy_encoded_text(self):
        """Copy encoded text to clipboard"""
        if self.encoded_text:
            self.master.clipboard_clear()
            self.master.clipboard_append(self.encoded_text)
            messagebox.showinfo("Copied", "Encoded text copied to clipboard")
    
    def save_encoded_text(self):
        """Save encoded text to a file"""
        if not self.encoded_text:
            messagebox.showerror("Error", "No encoded text to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save Encoded Text"
        )
        
        if file_path:
            success = export_encoded_text(self.encoded_text, file_path)
            if success:
                messagebox.showinfo("Success", f"Encoded text saved to {file_path}")
    
    def save_decoder_module(self):
        """Save standalone decoder as a Python module"""
        if not self.huffman_codes:
            messagebox.showerror("Error", "No Huffman codes to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".py",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")],
            title="Save Decoder Module"
        )
        
        if file_path:
            success = export_standalone_decoder(self.huffman_codes, file_path)
            if success:
                messagebox.showinfo("Success", f"Decoder module saved to {file_path}")
    
    def export_package(self):
        """Export a full package with all encoding data and tools"""
        if not self.encoded_text or not self.huffman_codes:
            messagebox.showerror("Error", "No encoding data to export")
            return
        
        folder_path = filedialog.askdirectory(
            title="Select Directory for Export Package"
        )
        
        if folder_path:
            package_name = os.path.join(folder_path, "huffman_package")
            success = export_full_package(
                self.input_text, self.encoded_text, self.huffman_codes,
                self.huffman_tree, self.compression_metrics, self.q_value,
                package_name
            )
            
            if success:
                messagebox.showinfo("Success", f"Full package exported to {package_name}")
    
    def test_decode(self):
        """Test decoding functionality"""
        if not self.huffman_codes:
            messagebox.showerror("Error", "No Huffman codes available for decoding")
            return
        
        encoded_input = self.decode_input.get(1.0, tk.END).strip()
        if not encoded_input:
            messagebox.showerror("Error", "Please enter encoded text to decode")
            return
        
        try:
            decoded_text = decode_text(encoded_input, self.huffman_codes)
            
            self.decode_result.config(state=tk.NORMAL)
            self.decode_result.delete(1.0, tk.END)
            self.decode_result.insert(tk.END, decoded_text)
            self.decode_result.config(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("Decoding Error", str(e))
            
            self.decode_result.config(state=tk.NORMAL)
            self.decode_result.delete(1.0, tk.END)
            self.decode_result.insert(tk.END, f"Error: {str(e)}")
            self.decode_result.config(state=tk.DISABLED)
    
    def export_frequency_data(self):
        """Export frequency data to a CSV file"""
        if not self.frequency_items:
            messagebox.showerror("Error", "No frequency data to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export Frequency Data"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', newline='') as f:
                    f.write("Symbol,Frequency,Percentage,Code\n")
                    for item in self.frequency_items:
                        symbol = item.symbol.replace(",", "\\,")  # Escape commas
                        code = self.huffman_codes.get(item.symbol, "")
                        f.write(f"{symbol},{item.frequency},{item.percentage:.2f},{code}\n")
                
                messagebox.showinfo("Success", f"Frequency data exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not export frequency data: {str(e)}")
    
    def export_performance_data(self):
        """Export Q-value performance data to a CSV file"""
        if not self.q_value_performance:
            messagebox.showerror("Error", "No performance data to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export Performance Data"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', newline='') as f:
                    f.write("Q Value,Compression Ratio,Bits Per Symbol,Tree Depth,Encoding Time (ms)\n")
                    for perf in self.q_value_performance:
                        f.write(f"{perf.q_value},{perf.compression_ratio:.2f},{perf.bits_per_symbol:.2f},{perf.tree_depth},{perf.encoding_time:.2f}\n")
                
                messagebox.showinfo("Success", f"Performance data exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not export performance data: {str(e)}")


# ==========================================================================
# Command-line Interface
# ==========================================================================

def cli_encode(input_file, output_file, q_value=2, verbose=False):
    """Command-line interface for encoding a file"""
    try:
        # Read input file
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if verbose:
            print(f"Read {len(text)} characters from {input_file}")
            print(f"Using Q-value: {q_value}")
        
        # Perform encoding
        start_time = time.time()
        
        frequencies = calculate_frequency(text)
        tree = build_qary_huffman_tree(frequencies, q_value)
        codes = assign_huffman_codes(tree)
        encoded_text = encode_text(text, codes)
        metrics = calculate_compression_metrics(text, encoded_text, codes, frequencies)
        
        end_time = time.time()
        encoding_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        if verbose:
            print(f"Encoding completed in {encoding_time:.2f} ms")
            print(f"Original size: {metrics.original_size} bits")
            print(f"Encoded size: {metrics.encoded_size} bits")
            print(f"Compression ratio: {metrics.compression_ratio:.2f}:1")
            print(f"Space savings: {metrics.space_savings} bytes ({metrics.percent_reduction:.1f}%)")
            print(f"Average bits per symbol: {metrics.bits_per_symbol:.2f}")
            print(f"Entropy: {metrics.entropy:.2f}")
            print(f"Encoding efficiency: {metrics.encoding_efficiency:.1f}%")
        
        # Write output to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(encoded_text)
        
        if verbose:
            print(f"Encoded output written to {output_file}")
        
        # Generate a decoder file
        decoder_file = output_file + ".decoder.py"
        export_standalone_decoder(codes, decoder_file)
        
        if verbose:
            print(f"Decoder module written to {decoder_file}")
        
        return True
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def cli_decode(input_file, decoder_file, output_file, verbose=False):
    """Command-line interface for decoding a file using a decoder module"""
    try:
        # Import the decoder module
        import importlib.util
        spec = importlib.util.spec_from_file_location("decoder", decoder_file)
        decoder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(decoder)
        
        # Read input file
        with open(input_file, 'r', encoding='utf-8') as f:
            encoded_text = f.read()
        
        if verbose:
            print(f"Read {len(encoded_text)} bits from {input_file}")
        
        # Perform decoding
        start_time = time.time()
        
        decoded_text = decoder.decode(encoded_text)
        
        end_time = time.time()
        decoding_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        if verbose:
            print(f"Decoding completed in {decoding_time:.2f} ms")
            print(f"Decoded {len(decoded_text)} characters")
        
        # Write output to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(decoded_text)
        
        if verbose:
            print(f"Decoded output written to {output_file}")
        
        return True
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

# ==========================================================================
# Main Application
# ==========================================================================

def main():
    """Main entry point for the application"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Q-ary Huffman Encoder')
    subparsers = parser.add_subparsers(dest='command', help='sub-command help')
    
    # GUI command
    gui_parser = subparsers.add_parser('gui', help='Start the graphical user interface')
    
    # Encode command
    encode_parser = subparsers.add_parser('encode', help='Encode a file')
    encode_parser.add_argument('input', help='Input file path')
    encode_parser.add_argument('output', help='Output file path')
    encode_parser.add_argument('-q', '--qvalue', type=int, default=2, help='Q value for encoding (default: 2)')
    encode_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    # Decode command
    decode_parser = subparsers.add_parser('decode', help='Decode a file')
    decode_parser.add_argument('input', help='Input file path (encoded text)')
    decode_parser.add_argument('decoder', help='Decoder module path')
    decode_parser.add_argument('output', help='Output file path')
    decode_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.command == 'encode':
        cli_encode(args.input, args.output, args.qvalue, args.verbose)
    
    elif args.command == 'decode':
        cli_decode(args.input, args.decoder, args.output, args.verbose)
    
    else:  # Default to GUI
        root = tk.Tk()
        root.title("Q-ary Huffman Encoder - by Haider Marouf")
        
        # Try to set a nicer theme if available
        try:
            import ttkthemes
            style = ttkthemes.ThemedStyle(root)
            style.set_theme("clam")  # Or use: "clam", "Adapta", "default", "classic"
        except ImportError:
            pass  # No theme, continue with default
        
        app = HuffmanEncoderApp(root)
        root.mainloop()

if __name__ == "__main__":
    main()
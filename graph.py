import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import csv
import numpy as np
import networkx as nx
import os # Adicionado para verificar a existência do arquivo pareto_front.csv

def load_scalar_from_setup(filename, key):
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' not in line:
                    continue
                current_key, value = line.split('=', 1)
                if current_key.strip() == key:
                    return value.strip()
    except FileNotFoundError:
        pass
    return None

def load_n_from_setup(filename):
    value = load_scalar_from_setup(filename, 'N')
    return int(value) if value is not None else None

N = load_n_from_setup('setup.temp')
FITNESS_MODEL = (load_scalar_from_setup('setup.temp', 'fitness_model') or 'linear').lower()

if N:
    print(f"Valor de N carregado: {N}")
else:
    print("Não foi possível encontrar o valor de N.")

def force_zero_diagonal(matrix):
    np.fill_diagonal(matrix, 0.0)
    return matrix

def linear_metric_label():
    if FITNESS_MODEL == "analitica":
        return "Fluxo Total da Rede"
    return "Desempenho do Sistema Linear (1 / |x|)"

def load_matrix_section(filename, section_name):
    matrix = []
    in_section = False

    try:
        with open(filename, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    if in_section and matrix:
                        break
                    continue

                if line.startswith("[") and line.endswith("]"):
                    current_section = line[1:-1]
                    if in_section:
                        break
                    in_section = current_section == section_name
                    continue

                if in_section:
                    matrix.append([float(value) for value in line.split()])
    except FileNotFoundError:
        return None
    except ValueError:
        return None

    if len(matrix) != N or any(len(row) != N for row in matrix):
        return None

    return force_zero_diagonal(np.array(matrix, dtype=float))

def load_matrix_from_csv(filepath, n, value_field="value"):
    matrix = np.zeros((n, n), dtype=float)
    seen = 0
    try:
        with open(filepath, newline='', encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                i = int(row["i"])
                j = int(row["j"])
                value = float(row[value_field])
                if 0 <= i < n and 0 <= j < n:
                    matrix[i, j] = value
                    seen += 1
    except (FileNotFoundError, ValueError, KeyError):
        return None

    if seen < n * n:
        return None
    return force_zero_diagonal(matrix)

def load_reference_matrix(filename):
    value_matrix = load_matrix_section(filename, "VALUE_MATRIX")
    if value_matrix is not None:
        return value_matrix

    min_matrix = load_matrix_section(filename, "MIN_MATRIX")
    max_matrix = load_matrix_section(filename, "MAX_MATRIX")
    if min_matrix is None or max_matrix is None:
        return load_matrix_from_csv("./files/value_matrix.csv", N)

    return force_zero_diagonal((min_matrix + max_matrix) / 2.0)

def load_history_data(filepath="./files/history_advanced_best_of_gen.csv"):
    generations, global_fitness, generational_fitness, genes_history = [], [], [], []
    try:
        with open(filepath, newline='', encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader, None)
            if not header:
                return generations, global_fitness, generational_fitness, genes_history

            try:
                generation_idx = header.index("Generation")
                global_linear_idx = header.index("GlobalBest_Linear")
                gen_linear_idx = header.index("GenBest_Linear")
                gene_start_idx = next(i for i, name in enumerate(header) if name.startswith("Gene_"))
            except (ValueError, StopIteration):
                return generations, global_fitness, generational_fitness, genes_history

            for row in reader:
                if len(row) < gene_start_idx:
                    continue
                generations.append(int(row[generation_idx]))
                global_fitness.append(float(row[global_linear_idx]))
                generational_fitness.append(float(row[gen_linear_idx]))
                genes_history.append([float(g) for g in row[gene_start_idx:]])
                
    except FileNotFoundError:
        print(f"Erro: O arquivo '{filepath}' não foi encontrado.")
    return generations, global_fitness, generational_fitness, genes_history

def load_b_vector(filepath="./files/b_vector.csv"):
    b_vector = []
    try:
        with open(filepath, newline='', encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                b_vector.append(float(row['value']))
    except FileNotFoundError:
        print(f"Erro: O arquivo '{filepath}' não foi encontrado.")
        return []
    except (ValueError, KeyError) as e:
        print(f"Erro ao processar o arquivo '{filepath}': {e}")
        return []
    return b_vector

def load_b_vector_from_setup(filename):
    in_section = False
    try:
        with open(filename, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    if in_section:
                        break
                    continue
                if line.startswith("[") and line.endswith("]"):
                    if in_section:
                        break
                    in_section = (line[1:-1] == "B_VECTOR")
                    continue
                if in_section:
                    values = [float(value) for value in line.split()]
                    if len(values) == N:
                        return values
                    return []
    except (FileNotFoundError, ValueError):
        return []
    return []

def collect_graph_data():
    generations, global_fitness, generational_fitness, genes_history = load_history_data()
    value_matrix = load_reference_matrix("setup.temp")
    b_vector = load_b_vector()
    if not b_vector:
        b_vector = load_b_vector_from_setup("setup.temp")

    missing = []
    if not generations:
        missing.append("historico")
    if value_matrix is None:
        missing.append("matriz de referencia")
    if not b_vector:
        missing.append("vetor b")

    return {
        "generations": generations,
        "global_fitness": global_fitness,
        "generational_fitness": generational_fitness,
        "genes_history": genes_history,
        "value_matrix": value_matrix,
        "b_vector": b_vector,
        "missing": missing,
    }

class GeneticAlgoAnalyzer(ttk.Frame):
    def __init__(self, parent, generations, global_fitness,
                 generational_fitness, genes_history,
                 value_matrix, b_vector):
        super().__init__(parent)

        self.generations = generations
        self.global_fitness = global_fitness
        self.generational_fitness = generational_fitness
        self.genes_history = genes_history
        self.graph_positions = None
        self.value_matrix = np.array(value_matrix, dtype=float) if value_matrix is not None else None
        self.b_vector = b_vector
        self.fitness_model = FITNESS_MODEL
        self.initial_solution_x = self._calculate_solution(0)
        self.active_nodes = [tk.BooleanVar(value=True) for _ in range(N)]

        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=1)
        self.canvas = tk.Canvas(main_frame)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.canvas_window_id = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.canvas.bind('<Configure>', self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        self._configure_grid()
        self._create_widgets()
        self._initial_display_update()

    def _on_canvas_configure(self, event):
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window_id, width=canvas_width)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def _configure_grid(self):
        self.scrollable_frame.grid_columnconfigure(0, weight=1)

    def _create_widgets(self):
        self._create_fitness_panel()
        self._create_pareto_panel()   # NOVO: Painel de Pareto integrado
        self._create_genes_panel()
        self._create_solution_panel()
        self._create_simulation_panel()
        self._create_graph_panel()

    def _create_fitness_panel(self):
        frame = ttk.LabelFrame(self.scrollable_frame, text="Evolução do Fitness", padding="10")
        frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=5)
        fig, ax = plt.subplots(figsize=(10, 5)) 
        ax.plot(self.generations, self.global_fitness, label="Melhor Fitness Global", color='blue')
        ax.plot(self.generations, self.generational_fitness, label="Fitness Geracional", color='red', alpha=0.7)
        ax.set_xlabel("Geração")
        ax.set_ylabel(linear_metric_label())
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _create_pareto_panel(self):
        frame = ttk.LabelFrame(self.scrollable_frame, text="Fronteira de Pareto (Trade-off de Objetivos)", padding="10")
        # Inserido na row=1
        frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        
        fig, ax = plt.subplots(figsize=(10, 4)) 
        
        linear_fits, path_fits = [], []
        filepath = "./files/pareto_front.csv"
        if not os.path.exists(filepath):
            filepath = "./files/pareto_front.csv"
            
        try:
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    linear_idx = header.index("Fitness_Linear") if header and "Fitness_Linear" in header else 1
                    path_idx = header.index("Fitness_Path") if header and "Fitness_Path" in header else 2
                    for row in reader:
                        if len(row) > max(linear_idx, path_idx):
                            linear_fits.append(float(row[linear_idx]))
                            path_fits.append(float(row[path_idx]))
        except Exception as e:
            print(f"Erro ao carregar Pareto no graph.py: {e}")

        if linear_fits:
            ax.scatter(linear_fits, path_fits, color='teal', edgecolors='black', s=50, alpha=0.85)
            
        ax.set_xlabel(linear_metric_label())
        ax.set_ylabel("Qualidade de Caminho")
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _create_genes_panel(self):
        frame = ttk.LabelFrame(self.scrollable_frame, text="Visualização dos Genes e Controles", padding="10")
        # Deslocado para row=2
        frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)

        controls_frame = ttk.Frame(frame)
        controls_frame.pack(pady=5, fill=tk.X)
        
        self.slider = tk.Scale(controls_frame, from_=0, to=len(self.generations)-1, orient=tk.HORIZONTAL,
                       label="Selecione a Geração")
        self.slider.bind("<ButtonRelease-1>", self.on_slider_change) # Só atualiza ao soltar o clique
        self.slider.pack(pady=5, fill=tk.X, expand=True)
        
        entry_frame = ttk.Frame(controls_frame)
        entry_frame.pack(pady=5)
        ttk.Label(entry_frame, text="Ou digite o Nº:").pack(side=tk.LEFT)
        self.entry_gen = ttk.Entry(entry_frame, width=10)
        self.entry_gen.pack(side=tk.LEFT, padx=5)
        ttk.Button(entry_frame, text="Atualizar", command=self.on_manual_entry).pack(side=tk.LEFT)
        
        self.genes_text = tk.Label(frame, text="", justify=tk.CENTER, font=("Courier", 11))
        self.genes_text.pack(pady=20, fill=tk.BOTH, expand=True)

    def _create_solution_panel(self):
        frame = ttk.LabelFrame(self.scrollable_frame, text="Comparação da Solução do Vetor X", padding="10")
        # Deslocado para row=3
        frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=5)
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        columns = ('node', 'gen0', 'gen_curr', 'abs_diff', 'perc_diff')
        self.solution_tree = ttk.Treeview(frame, columns=columns, show='headings', height=N)
        
        self.solution_tree.heading('node', text='Nó (i)')
        self.solution_tree.heading('gen0', text='Solução Geração 0')
        self.solution_tree.heading('gen_curr', text='Solução Geração Atual')
        self.solution_tree.heading('abs_diff', text='Dif. Absoluta')
        self.solution_tree.heading('perc_diff', text='Dif. %')
        
        self.solution_tree.column('node', width=80, anchor='center')
        self.solution_tree.column('gen0', width=150, anchor='center')
        self.solution_tree.column('gen_curr', width=160, anchor='center')
        self.solution_tree.column('abs_diff', width=140, anchor='center')
        self.solution_tree.column('perc_diff', width=120, anchor='center')

        tree_scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.solution_tree.yview)
        self.solution_tree.configure(yscrollcommand=tree_scrollbar.set)
        
        self.solution_tree.grid(row=0, column=0, sticky='nsew')
        tree_scrollbar.grid(row=0, column=1, sticky='ns')

    def _create_simulation_panel(self):
        frame = ttk.LabelFrame(self.scrollable_frame, text="Simulação de Desligamento de Nós", padding="10")
        # Deslocado para row=4
        frame.grid(row=4, column=0, sticky="nsew", padx=10, pady=5)

        center_wrapper_frame = ttk.Frame(frame)
        center_wrapper_frame.pack(pady=5, fill="x")

        checkbox_frame = ttk.Frame(center_wrapper_frame)
        checkbox_frame.pack()
        
        cols = 10 
        for i in range(N):
            var = self.active_nodes[i]
            cb = ttk.Checkbutton(checkbox_frame, text=str(i), variable=var, command=self.update_simulation_fitness)
            cb.grid(row=i // cols, column=i % cols, padx=5, pady=2, sticky="w")
        
        self.simulation_fitness_label = ttk.Label(frame, text="Fitness Simulado: -", font=("Arial", 14, "bold"))
        self.simulation_fitness_label.pack(pady=15)

    def _create_graph_panel(self):
        frame = ttk.LabelFrame(self.scrollable_frame, text="Visualização do Grafo de Conexões (Verde = Bom Fitness/Influência)", padding="10")
        # Deslocado para row=5
        frame.grid(row=5, column=0, sticky="nsew", padx=10, pady=5)
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        self.graph_canvas = tk.Canvas(frame, bg='white', height=550)
        self.graph_canvas.grid(row=0, column=0, sticky="nsew")

        self.node_coords = {}
        self.node_items = {}
        self.edge_items = {}
        self.adjacency_matrix = []
        
        # Dicionários para armazenar a cor real (fitness) de cada elemento
        self.node_base_colors = {}
        self.edge_base_colors = {}

    def _initial_display_update(self):
        if self.generations:
            initial_idx = len(self.generations) - 1
            self.slider.set(initial_idx)
            self.entry_gen.insert(0, str(self.generations[initial_idx]))
            self.after(100, lambda: self.update_displays(initial_idx))
    
    def calculate_fitness_py(self, A_matrix, b_vector):
        sol = self._solve_for_current_model(np.array(A_matrix, dtype=float), np.array(b_vector, dtype=float))
        if sol is None:
            return 0.0
        if self.fitness_model == "analitica":
            return float(np.sum(sol))
        total_abs = np.sum(np.abs(sol))
        if total_abs == 0.0:
            return 0.0
        return 1.0 / total_abs

    def _infer_uniform_c(self, A_matrix):
        upper_values = A_matrix[np.triu_indices(N, k=1)]
        positive_values = upper_values[upper_values > 1e-12]
        if positive_values.size == 0:
            diag_values = np.diag(A_matrix)
            positive_values = diag_values[diag_values > 1e-12]
        if positive_values.size == 0:
            return 0.0
        return float(np.mean(positive_values))

    def _solve_for_current_model(self, A_matrix, b_vector):
        if self.fitness_model == "analitica":
            if N <= 1:
                return np.zeros_like(b_vector, dtype=float)
            c = self._infer_uniform_c(A_matrix)
            if c <= 1e-12:
                return None
            s_total = float(np.sum(b_vector))
            term = s_total / (N - 1.0)
            return (term - b_vector) / c

        try:
            return np.linalg.solve(A_matrix, b_vector)
        except np.linalg.LinAlgError:
            return None

    def update_simulation_fitness(self):
        gen_index = self.slider.get()
        if self.value_matrix is None or gen_index >= len(self.genes_history):
            self.simulation_fitness_label.config(text="Fitness Simulado: Erro")
            return

        genes_flat = self.genes_history[gen_index]
        positions = np.array([genes_flat[i:i+N] for i in range(0, len(genes_flat), N)])

        A_sim = positions * self.value_matrix
        force_zero_diagonal(A_sim)
        b_sim = np.array(self.b_vector, dtype=float)
        for i in range(N):
            if not self.active_nodes[i].get():
                A_sim[i, :] = 0.0
                A_sim[:, i] = 0.0
                b_sim[i] = 0.0
                A_sim[i][i] = 0.0
        
        fitness_score = self.calculate_fitness_py(A_sim, b_sim)
        self.simulation_fitness_label.config(text=f"Fitness Simulado: {fitness_score:.4f}")

        self._update_graph_node_colors()

    def _calculate_solution(self, generation_index):
        if self.value_matrix is None or generation_index >= len(self.genes_history):
            return None
        genes_flat = self.genes_history[generation_index]
        positions = np.array([genes_flat[i:i+N] for i in range(0, len(genes_flat), N)])
        A = positions * self.value_matrix
        force_zero_diagonal(A)
        return self._solve_for_current_model(A, np.array(self.b_vector, dtype=float))

    def format_genes(self, genes_list):
        return "\n".join(" ".join(map(str, genes_list[i:i+N])) for i in range(0, len(genes_list), N))

    def update_displays(self, generation_index):
        self.update_genes_display(generation_index)
        self.update_solution_display(generation_index)
        self.update_simulation_panel()
        self.update_graph_display(generation_index)
        self.scrollable_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def update_genes_display(self, generation_index):
        genes_str = self.format_genes(self.genes_history[generation_index])
        self.genes_text.config(text=genes_str)

    def update_simulation_panel(self):
        for var in self.active_nodes:
            var.set(True)
        self.update_simulation_fitness()

    def update_solution_display(self, generation_index):
        for item in self.solution_tree.get_children():
            self.solution_tree.delete(item)
        
        current_solution_x = self._calculate_solution(generation_index)
        
        if self.initial_solution_x is None or current_solution_x is None:
            msg = "Solução inicial não pôde ser calculada." if self.initial_solution_x is None else "Solução atual não pôde ser calculada (matriz singular)."
            self.solution_tree.insert("", "end", values=("Erro", msg, "", "", ""))
            return

        for i in range(N):
            initial_val = self.initial_solution_x[i]
            current_val = current_solution_x[i]
            abs_diff = abs(current_val - initial_val)
            
            if initial_val != 0:
                perc_diff = ((current_val - initial_val) / abs(initial_val)) * 100
                perc_str = f"{perc_diff:+.2f}%"
            else:
                perc_str = "N/A"

            self.solution_tree.insert("", "end", values=(
                f"x[{i}]", f"{initial_val:9.4f}", f"{current_val:9.4f}",
                f"{abs_diff:9.4f}", perc_str
            ))
            
    def get_color_from_value(self, val, vmin, vmax):
        if vmax == vmin:
            norm = 0.5
        else:
            norm = (val - vmin) / (vmax - vmin)
        norm = max(0.0, min(1.0, norm))
        rgba = cm.RdYlGn(norm)
        return mcolors.to_hex(rgba)

    def update_graph_display(self, generation_index):
        self.graph_canvas.delete("all")
        
        genes_flat = self.genes_history[generation_index]
        self.adjacency_matrix = [genes_flat[i:i+N] for i in range(0, len(genes_flat), N)]

        positions = np.array(self.adjacency_matrix)
        A_base = positions * self.value_matrix
        force_zero_diagonal(A_base)
        b_base = np.array(self.b_vector, dtype=float)
        
        base_fitness = self.calculate_fitness_py(A_base, b_base)
        
        node_impacts = {}
        for i in range(N):
            A_sim = A_base.copy()
            b_sim = b_base.copy()
            A_sim[i, :] = 0.0
            A_sim[:, i] = 0.0
            b_sim[i] = 0.0
            A_sim[i][i] = 0.0
            fit = self.calculate_fitness_py(A_sim, b_sim)
            node_impacts[i] = base_fitness - fit  

        edge_impacts = {}
        for i in range(N):
            for j in range(i + 1, N):
                if self.adjacency_matrix[i][j] == 1:
                    A_sim = A_base.copy()
                    A_sim[i, j] = 0.0
                    A_sim[j, i] = 0.0
                    fit = self.calculate_fitness_py(A_sim, b_base)
                    edge_impacts[(i, j)] = base_fitness - fit

        min_node = min(node_impacts.values()) if node_impacts else 0
        max_node = max(node_impacts.values()) if node_impacts else 0
        self.node_base_colors = {i: self.get_color_from_value(node_impacts[i], min_node, max_node) for i in range(N)}
        
        min_edge = min(edge_impacts.values()) if edge_impacts else 0
        max_edge = max(edge_impacts.values()) if edge_impacts else 0
        self.edge_base_colors = {e: self.get_color_from_value(edge_impacts[e], min_edge, max_edge) for e in edge_impacts}

        canvas_width = self.graph_canvas.winfo_width()
        canvas_height = self.graph_canvas.winfo_height()
        
        padding = 50 
        drawable_width = canvas_width - 2 * padding
        drawable_height = canvas_height - 2 * padding

        self.node_coords.clear()
        self.node_items.clear()
        self.edge_items.clear()

        G = nx.Graph()
        G.add_nodes_from(range(N))
        for i in range(N):
            for j in range(i + 1, N):
                if self.adjacency_matrix[i][j] == 1:
                    G.add_edge(i, j)
        
       # Configura o layout do grafo apenas na primeira vez para não travar a UI
        if self.graph_positions is None:
            G_temp = nx.Graph()
            G_temp.add_nodes_from(range(N))
            # Cria um layout circular ou usa spring_layout num grafo totalmente conectado
            self.graph_positions = nx.circular_layout(G_temp) # Muito mais rápido e limpo!
        
        pos = self.graph_positions
        
        xs = [coords[0] for coords in pos.values()]
        ys = [coords[1] for coords in pos.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        for i in range(N):
            norm_x, norm_y = pos[i]
            
            if max_x != min_x:
                scaled_x = (norm_x - min_x) / (max_x - min_x)
            else:
                scaled_x = 0.5
                
            if max_y != min_y:
                scaled_y = (norm_y - min_y) / (max_y - min_y)
            else:
                scaled_y = 0.5

            final_x = padding + scaled_x * drawable_width
            final_y = padding + scaled_y * drawable_height
            
            self.node_coords[i] = (final_x, final_y)

        for i in range(N):
            for j in range(i + 1, N):
                if self.adjacency_matrix[i][j] == 1:
                    x1, y1 = self.node_coords[i]
                    x2, y2 = self.node_coords[j]
                    color = self.edge_base_colors.get((i, j), 'lightgrey')
                    edge = self.graph_canvas.create_line(x1, y1, x2, y2, fill=color, width=2.0)
                    self.edge_items[(i, j)] = edge

        node_radius = 12
        for i in range(N):
            x, y = self.node_coords[i]
            node_tag = f"node_{i}"
            color = self.node_base_colors.get(i, 'skyblue')
            oval = self.graph_canvas.create_oval(
                x - node_radius, y - node_radius, x + node_radius, y + node_radius,
                fill=color, outline='black', width=1.5, tags=node_tag
            )
            self.graph_canvas.create_text(x, y, text=str(i), font=("Arial", 8, "bold"), tags=node_tag)
            self.node_items[i] = oval
            self.graph_canvas.tag_bind(node_tag, '<Enter>', lambda e, node_id=i: self._on_node_enter(node_id))
            self.graph_canvas.tag_bind(node_tag, '<Leave>', self._on_node_leave)

    def _update_graph_node_colors(self):
        for i in self.node_items:
            node_item = self.node_items[i]
            if self.active_nodes[i].get():
                color = self.node_base_colors.get(i, 'skyblue')
                self.graph_canvas.itemconfig(node_item, fill=color, outline='black')
            else:
                self.graph_canvas.itemconfig(node_item, fill='grey', outline='darkgrey')

    def _on_node_enter(self, node_id):
        if not self.active_nodes[node_id].get():
            return

        for i in self.node_items:
            self.graph_canvas.itemconfig(self.node_items[i], fill='#f0f0f0', outline='lightgrey')
        for edge in self.edge_items.values():
            self.graph_canvas.itemconfig(edge, fill='#f0f0f0')

        self.graph_canvas.itemconfig(self.node_items[node_id], fill='blue', outline='black')

        for neighbor_id in range(N):
            if self.adjacency_matrix[node_id][neighbor_id] == 1 and node_id != neighbor_id:
                edge_key = tuple(sorted((node_id, neighbor_id)))
                if edge_key in self.edge_items:
                    self.graph_canvas.itemconfig(self.edge_items[edge_key], fill='black', width=2.5)
                
                self.graph_canvas.itemconfig(self.node_items[neighbor_id], fill='cyan', outline='black')
                
    def _on_node_leave(self, event):
        for edge_key, edge in self.edge_items.items():
            color = self.edge_base_colors.get(edge_key, 'lightgrey')
            self.graph_canvas.itemconfig(edge, fill=color, width=2.0)
        
        self._update_graph_node_colors()

    def on_slider_change(self, event=None):
        generation_idx = int(self.slider.get())
        self.update_displays(generation_idx)
        self.entry_gen.delete(0, tk.END)
        self.entry_gen.insert(0, str(self.generations[generation_idx]))

    def on_manual_entry(self):
        try:
            gen_num = int(self.entry_gen.get())
            if gen_num in self.generations:
                gen_idx = self.generations.index(gen_num)
                self.slider.set(gen_idx)
        except ValueError:
            print("Entrada manual inválida.")

def build_graph_frame(parent):
    data = collect_graph_data()
    if data["missing"]:
        raise RuntimeError("Dados insuficientes para carregar gráficos: " + ", ".join(data["missing"]))

    frame = GeneticAlgoAnalyzer(
        parent,
        data["generations"],
        data["global_fitness"],
        data["generational_fitness"],
        data["genes_history"],
        data["value_matrix"],
        data["b_vector"]
    )
    frame.pack(fill="both", expand=True)
    return frame

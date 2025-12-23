import customtkinter as ctk
import subprocess
import threading
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")
class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Interface de Treinamento e Monitoramento")
        self.geometry("1100x800")

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        self.left_container = ctk.CTkFrame(self)
        self.left_container.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Título fixo no topo
        self.label_title = ctk.CTkLabel(self.left_container, text="Configurações (setup.temp)", font=("Roboto", 20, "bold"))
        self.label_title.pack(pady=10)

        # --- Área de Scroll para os Inputs (Para caber as matrizes) ---
        self.scroll_frame = ctk.CTkScrollableFrame(self.left_container, label_text="Parâmetros")
        self.scroll_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Dicionários de armazenamento
        self.scalar_inputs = {}
        self.matrix_inputs = {}
        self.vector_inputs = {}

        # 1. Cria inputs escalares (N, GEN, POP_SIZE...)
        self.create_scalar_inputs()

        # 2. Botão para atualizar grades baseado em N inside do scroll
        self.btn_update_grid = ctk.CTkButton(self.scroll_frame, text="Gerar Grades (Baseado em N)", 
                                             command=self.generate_matrix_grids, fg_color="#3B8ED0")
        self.btn_update_grid.pack(pady=10, fill="x")

        # 3. Container para Matrizes e Vetores
        self.matrices_container = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        self.matrices_container.pack(fill="both", expand=True)

        # --- Área de Botões Fixos (Rodapé do lado esquerdo) ---
        self.controls_frame = ctk.CTkFrame(self.left_container, fg_color="transparent")
        self.controls_frame.pack(fill="x", padx=5, pady=10)

        # Botão Salvar
        self.btn_save = ctk.CTkButton(self.controls_frame, text="SALVAR ARQUIVO", command=self.save_setup, fg_color="gray")
        self.btn_save.pack(pady=5, fill="x")

        # Botão Iniciar
        self.btn_start = ctk.CTkButton(self.controls_frame, text="INICIAR TREINAMENTO", command=self.start_process, fg_color="green")
        self.btn_start.pack(pady=5, fill="x")

        # Botão Cancelar
        self.btn_stop = ctk.CTkButton(self.controls_frame, text="CANCELAR", command=self.stop_process, fg_color="red", state="disabled")
        self.btn_stop.pack(pady=5, fill="x")

        # ===================================================
        # --- LADO DIREITO (Gráfico e Logs) ---
        # ===================================================
        self.right_frame = ctk.CTkFrame(self)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Evolução do Treinamento")
        self.ax.set_xlabel("Época / Tempo")
        self.ax.set_ylabel("Valor / Loss")
        self.line, = self.ax.plot([], [], 'r-') 

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.right_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        self.log_box = ctk.CTkTextbox(self.right_frame, height=150)
        self.log_box.pack(fill="x", padx=10, pady=10)
        self.log_box.insert("0.0", "Aguardando configuração...\n")

        self.process = None
        self.is_running = False
        self.x_data = []
        self.y_data = []

        # Tenta carregar dados existentes ao iniciar o app
        self.load_existing_setup()
        self.plot_update_counter = 0  # Novo contador
        self.plot_update_interval = 2 # Atualiza o gráfico a cada 10 dados recebidos (ajuste conforme necessário)
    # ===================================================
    # --- MÉTODOS DE INPUT E ARQUIVO ---
    # ===================================================
    def update_graph(self, x, y):
        # 1. Apenas adiciona os dados (operação muito rápida)
        self.x_data.append(float(x))
        self.y_data.append(float(y))
        
        self.plot_update_counter += 1

        # 2. Só redesenha se atingir o intervalo
        if self.plot_update_counter >= self.plot_update_interval:
            self.line.set_data(self.x_data, self.y_data)
            self.ax.relim()
            self.ax.autoscale_view()
            self.canvas.draw()
            self.plot_update_counter = 0 # Reseta o contador
    def create_scalar_inputs(self):
        fields = [
            ("N", "3"), ("POP_SIZE", "100"), ("GEN", "1000"), ("MU_TAX_BASE", "0.01"),
            ("TOURNAMENT_SIZE", "20"), ("EVAL_MATRICES", "20"), 
            ("EVAL_LOOPS", "20"), ("REGEN_INTERVAL", "20")
        ]
        
        for label_text, default_val in fields:
            lbl = ctk.CTkLabel(self.scroll_frame, text=label_text, anchor="w")
            lbl.pack(pady=(5, 0), padx=5, fill="x")
            
            entry = ctk.CTkEntry(self.scroll_frame)
            entry.insert(0, default_val)
            entry.pack(pady=(0, 5), padx=5, fill="x")
            
            self.scalar_inputs[label_text] = entry

    def generate_matrix_grids(self):
        for widget in self.matrices_container.winfo_children():
            widget.destroy()
        
        self.matrix_inputs = {}
        self.vector_inputs = {}

        try:
            n_val = int(self.scalar_inputs["N"].get())
        except ValueError:
            self.log_message("Erro: N deve ser um número inteiro.")
            return

        matrices_sections = ["MIN_MATRIX", "MAX_MATRIX", "INITIAL_POSITIONS"]
        vector_sections = ["MAX_CONNECTIONS", "B_VECTOR"]

        for section in matrices_sections:
            self.create_grid_section(section, n_val, n_val, is_vector=False)

        for section in vector_sections:
            self.create_grid_section(section, 1, n_val, is_vector=True)

    def create_grid_section(self, title, rows, cols, is_vector):
        frame = ctk.CTkFrame(self.matrices_container)
        frame.pack(pady=10, fill="x")

        lbl = ctk.CTkLabel(frame, text=f"[{title}]", font=("Arial", 12, "bold"))
        lbl.pack(pady=5)

        grid_frame = ctk.CTkFrame(frame, fg_color="transparent")
        grid_frame.pack()

        entry_list = []
        for r in range(rows):
            row_entries = []
            for c in range(cols):
                entry = ctk.CTkEntry(grid_frame, width=40, height=25)
                entry.grid(row=r, column=c, padx=2, pady=2)
                row_entries.append(entry)
            
            if is_vector:
                entry_list = row_entries
            else:
                entry_list.append(row_entries)

        if is_vector:
            self.vector_inputs[title] = entry_list
        else:
            self.matrix_inputs[title] = entry_list

    def save_setup(self):
        try:
            with open("setup.temp", "w") as f:
                for key, entry in self.scalar_inputs.items():
                    f.write(f"{key}={entry.get()}\n")
                f.write("\n")

                for key, rows in self.matrix_inputs.items():
                    f.write(f"[{key}]\n")
                    for row in rows:
                        values = [e.get() for e in row]
                        f.write(" ".join(values) + "\n")
                    f.write("\n")

                for key, cols in self.vector_inputs.items():
                    f.write(f"[{key}]\n")
                    values = [e.get() for e in cols]
                    f.write(" ".join(values) + "\n")
                    f.write("\n")
            
            self.log_message("Arquivo setup.temp salvo com sucesso!")
            return True
        except Exception as e:
            self.log_message(f"Erro ao salvar: {e}")
            return False

    def load_existing_setup(self):
        if not os.path.exists("setup.temp"):
            self.generate_matrix_grids()
            return

        try:
            with open("setup.temp", "r") as f:
                lines = f.readlines()

            data_scalars = {}
            current_section = None
            matrix_buffer = []
            parsed_matrices = {} 
            
            for line in lines:
                line = line.strip()
                if not line: continue

                if "=" in line and current_section is None:
                    k, v = line.split("=")
                    data_scalars[k.strip()] = v.strip()
                elif line.startswith("[") and line.endswith("]"):
                    if current_section:
                        parsed_matrices[current_section] = matrix_buffer
                    current_section = line[1:-1]
                    matrix_buffer = []
                else:
                    if current_section:
                        matrix_buffer.append(line.split())

            if current_section:
                parsed_matrices[current_section] = matrix_buffer

            for k, v in data_scalars.items():
                if k in self.scalar_inputs:
                    self.scalar_inputs[k].delete(0, "end")
                    self.scalar_inputs[k].insert(0, v)

            self.generate_matrix_grids()

            for key, grid_widgets in self.matrix_inputs.items():
                if key in parsed_matrices:
                    data_rows = parsed_matrices[key]
                    for r, row_widgets in enumerate(grid_widgets):
                        if r < len(data_rows):
                            for c, widget in enumerate(row_widgets):
                                if c < len(data_rows[r]):
                                    widget.delete(0, "end")
                                    widget.insert(0, data_rows[r][c])

            for key, grid_widgets in self.vector_inputs.items():
                if key in parsed_matrices and len(parsed_matrices[key]) > 0:
                    data_vals = parsed_matrices[key][0]
                    for c, widget in enumerate(grid_widgets):
                        if c < len(data_vals):
                            widget.delete(0, "end")
                            widget.insert(0, data_vals[c])
            
            self.log_message("setup.temp carregado.")

        except Exception as e:
            self.log_message(f"Erro ao ler arquivo: {e}")


    def start_process(self):
        if self.is_running:
            return

        # Salva automaticamente antes de começar para garantir que o .exe leia o mais recente
        if not self.save_setup():
            return 

        self.x_data = []
        self.y_data = []
        self.line.set_data([], [])
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()
        self.log_box.delete("0.0", "end")
        self.log_message("Iniciando subprocesso...")

        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.is_running = True

        # Inicia thread passando o nome do arquivo se necessário, ou vazio
        self.thread = threading.Thread(target=self.run_async_code, args=("setup.temp",))
        self.thread.start()

    def stop_process(self):
        if self.process and self.is_running:
            self.process.terminate()
            self.log_message("Processo cancelado pelo usuário.")
            self.cleanup_state()

    def cleanup_state(self):
        self.is_running = False
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")

    def log_message(self, message):
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end") 

    def update_graph(self, x, y):
        self.x_data.append(float(x))
        self.y_data.append(float(y))
        
        self.line.set_data(self.x_data, self.y_data)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

    def run_async_code(self, param):
        cmd = ["./genetic_solver.exe", ""] 
        # Se estiver no Windows, as vezes é bom forçar o buffer off ou usar stdbuf no linux, 
        # mas aqui vamos focar no Python.

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1 # Line buffered
            )

            while True:
                # O readline pode bloquear se o subprocesso não der flush
                line = self.process.stdout.readline()
                
                if not line and self.process.poll() is not None:
                    break
                
                if line:
                    line = line.strip()
                    if line.startswith("Dados:"):
                        try:
                            _, dados = line.split(":")
                            x, y = dados.split(",")
                            self.after(0, self.update_graph, x, y)
                        except ValueError:
                            pass
                    else:
                        # Logs normais (que não são dados brutos) podem ir para a caixa
                        self.after(0, self.log_message, line)

            rc = self.process.poll()
            self.after(0, self.log_message, f"Processo finalizado com código {rc}")
            
            # Garante um desenho final para pegar os dados que sobraram no contador
            self.after(0, self.force_final_draw)

        except Exception as e:
            self.after(0, self.log_message, f"Erro: {str(e)}")
        finally:
            self.after(0, self.cleanup_state)

    # Crie este método auxiliar para desenhar o resto dos dados ao final
    def force_final_draw(self):
        self.line.set_data(self.x_data, self.y_data)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()
if __name__ == "__main__":
    app = App()
    app.mainloop()
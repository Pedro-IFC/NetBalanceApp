import customtkinter as ctk
import subprocess
import threading
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import graph

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
        
        self.label_title = ctk.CTkLabel(self.left_container, text="Configurações (setup.temp)", font=("Roboto", 20, "bold"))
        self.label_title.pack(pady=10)

        self.scroll_frame = ctk.CTkScrollableFrame(self.left_container, label_text="Parâmetros")
        self.scroll_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.scalar_inputs = {}
        # NOVO: Dicionário para armazenar as matrizes apenas na memória, sem renderizar
        self.in_memory_matrices = {}

        self.create_scalar_inputs()

        # REMOVIDO: Botão de Gerar Grades e o container de matrizes

        self.controls_frame = ctk.CTkFrame(self.left_container, fg_color="transparent")
        self.controls_frame.pack(fill="x", padx=5, pady=10)

        self.btn_save = ctk.CTkButton(self.controls_frame, text="SALVAR ARQUIVO", command=self.save_setup, fg_color="gray")
        self.btn_save.pack(pady=5, fill="x")

        self.btn_start = ctk.CTkButton(self.controls_frame, text="INICIAR TREINAMENTO", command=self.start_process, fg_color="green")
        self.btn_start.pack(pady=5, fill="x")

        self.btn_stop = ctk.CTkButton(self.controls_frame, text="CANCELAR", command=self.stop_process, fg_color="red", state="disabled")
        self.btn_stop.pack(pady=5, fill="x")

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

        self.load_existing_setup()
        self.plot_update_counter = 0
        self.plot_update_interval = 2
        
    def load_full_graph_view(self):
        for widget in self.right_frame.winfo_children():
            widget.destroy()

        graph_container = ctk.CTkFrame(self.right_frame)
        graph_container.pack(fill="both", expand=True)

        try:
            graph.build_graph_frame(graph_container)
        except Exception as e:
            msg = str(e)
            label = ctk.CTkLabel(
                graph_container,
                text=f"Não foi possível carregar a visualização final.\n{msg}",
                justify="left"
            )
            label.pack(padx=20, pady=20, anchor="w")
            self.log_message(f"Erro ao carregar visualização final: {msg}")

    def update_graph(self, x, y):
        self.x_data.append(float(x))
        self.y_data.append(float(y))
        
        self.plot_update_counter += 1

        if self.plot_update_counter >= self.plot_update_interval:
            self.line.set_data(self.x_data, self.y_data)
            self.ax.relim()
            self.ax.autoscale_view()
            self.canvas.draw()
            self.plot_update_counter = 0 

    def create_scalar_inputs(self):
        fields = [
            ("N", "3"), ("POP_SIZE", "100"), ("GEN", "1000"), ("MU_TAX_BASE", "0.01"),
            ("TOURNAMENT_SIZE", "20"), ("EVAL_MATRICES", "20"), 
            ("EVAL_LOOPS", "20"), ("REGEN_INTERVAL", "20"),
            ("fitness_model", "analitica")
        ]
        
        for label_text, default_val in fields:
            lbl = ctk.CTkLabel(self.scroll_frame, text=label_text, anchor="w")
            lbl.pack(pady=(5, 0), padx=5, fill="x")
            
            entry = ctk.CTkEntry(self.scroll_frame)
            entry.insert(0, default_val)
            entry.pack(pady=(0, 5), padx=5, fill="x")
            
            self.scalar_inputs[label_text] = entry

    # REMOVIDO: generate_matrix_grids e create_grid_section

    def save_setup(self):
        try:
            with open("setup.temp", "w") as f:
                # 1. Salva os escalares editados na UI
                for key, entry in self.scalar_inputs.items():
                    f.write(f"{key}={entry.get()}\n")
                f.write("\n")

                # 2. Salva as matrizes que estavam na memória de volta no arquivo
                for key, rows in self.in_memory_matrices.items():
                    f.write(f"[{key}]\n")
                    for row in rows:
                        f.write(" ".join(row) + "\n")
                    f.write("\n")
            
            self.log_message("Arquivo setup.temp salvo com sucesso!")
            return True
        except Exception as e:
            self.log_message(f"Erro ao salvar: {e}")
            return False

    def load_existing_setup(self):
        if not os.path.exists("setup.temp"):
            self.log_message("Arquivo setup.temp não encontrado. Operando apenas com escalares padrão.")
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

            # Atualiza os inputs escalares visuais
            for k, v in data_scalars.items():
                if k in self.scalar_inputs:
                    self.scalar_inputs[k].delete(0, "end")
                    self.scalar_inputs[k].insert(0, v)

            # Guarda as matrizes lidas na memória para salvar depois
            self.in_memory_matrices = parsed_matrices
            
            self.log_message("setup.temp carregado (matrizes salvas em memória).")

        except Exception as e:
            self.log_message(f"Erro ao ler arquivo: {e}")

    def start_process(self):
        if self.is_running:
            return

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

    def run_async_code(self, param):
        cmd = ["./genetic_solver.exe", ""] 

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            while True:
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
                        self.after(0, self.log_message, line)

            rc = self.process.poll()
            self.after(0, self.log_message, f"Processo finalizado com código {rc}")
            
            self.after(0, self.force_final_draw)

        except Exception as e:
            self.after(0, self.log_message, f"Erro: {str(e)}")
        finally:
            self.after(0, self.cleanup_state)
            
        self.after(0, self.load_full_graph_view)

    def force_final_draw(self):
        self.line.set_data(self.x_data, self.y_data)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

if __name__ == "__main__":
    app = App()
    app.mainloop()

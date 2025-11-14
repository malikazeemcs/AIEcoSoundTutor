import time
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import ttkbootstrap as tb
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import io
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
import threading
import os
import soundfile as sf
from scipy.signal import spectrogram
import mplcursors
# import simpleaudio as sa
import pandas as pd
from sklearn.metrics import silhouette_score, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import librosa
import openl3
# from Tooltip import ToolTip
import sounddevice as sd

# --- Helper Functions (unchanged from previous version) ---
def reduce_dim(features, method="PCA", n_components=2):
    """
    Reduces the dimensionality of features using PCA, t-SNE, or UMAP.
    Preserves 'second', 'file_name', and 'label' columns.
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import umap.umap_ as umap

    features_copy = features.copy()

    special_cols = ['second', 'file_name']
    if 'label' in features_copy.columns:
        special_cols.append('label')

    feature_columns = [col for col in features_copy.columns if col not in special_cols]

    if method == "PCA":
        reducer = PCA(n_components=n_components)
    elif method == "t-SNE":
        perp = 30
        if feature_columns.__len__()<perp:
            perp = len(feature_columns)

        reducer = TSNE(n_components=n_components, random_state=42, perplexity=perp)
    elif method == "UMAP":
        reducer = umap.UMAP(n_components=n_components, random_state=42)
    else:
        raise ValueError("Invalid dimensionality reduction method.")

    reduced = reducer.fit_transform(features_copy[feature_columns])
    df_reduced = pd.DataFrame(reduced, columns=['x', 'y'])

    for col in special_cols:
        if col in features_copy.columns:
            df_reduced[col] = features_copy[col].values

    return df_reduced


def apply_clustering(features, algorithm="KMeans", n_clusters=2):
    """
    Applies clustering to features and returns labels and silhouette score.
    """
    from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
    from sklearn.mixture import GaussianMixture
    import hdbscan

    cluster_data = features[['x', 'y']]
    labels = None
    s_score = 0.0

    if algorithm == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        labels = model.fit_predict(cluster_data)
    elif algorithm == "GMM":
        model = GaussianMixture(n_components=n_clusters, random_state=42)
        labels = model.fit_predict(cluster_data)
    elif algorithm == "HDBSCAN":
        model = hdbscan.HDBSCAN(min_cluster_size=3)
        labels = model.fit_predict(cluster_data)
    else:
        raise ValueError("Invalid clustering algorithm.")

    unique_labels = np.unique(labels)
    if len(unique_labels) > 1 and len(unique_labels) < len(labels):
        s_score = silhouette_score(cluster_data, labels)

    return labels, s_score, len(unique_labels)


def extract_mfcc(audio_file, sr=22050, n_mfcc=13, segment_length=1.0):
    """
    Extracts MFCC features from an audio file in segments.
    """
    if not (0.1 <= segment_length <= 60.0):
        segment_length = 1.0
    y, sr = librosa.load(audio_file, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)
    mfccs = []
    start_times = []
    current_time = 0.0
    while current_time + segment_length <= duration:
        start_sample = int(current_time * sr)
        end_sample = start_sample + int(segment_length * sr)
        y_segment = y[start_sample:end_sample]
        if len(y_segment) == int(segment_length * sr):
            mfcc = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=n_mfcc)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfccs.append(mfcc_mean)
            start_times.append(current_time)
        current_time += segment_length
    df = pd.DataFrame(mfccs, columns=[f"mfcc_{i + 1}" for i in range(n_mfcc)])
    df["second"] = start_times
    return df


def extract_openl3(audio_file, segment_length=1.0):
    """
    Extracts OpenL3 embeddings from an audio file.
    """
    audio, sr = sf.read(audio_file)
    emb, ts = openl3.get_audio_embedding(
        audio,
        sr,
        input_repr="mel256",
        content_type="env",
        embedding_size=512,
        hop_size=segment_length
    )
    emb_mean = pd.DataFrame(emb)
    emb_mean["second"] = np.arange(len(emb_mean)) * segment_length
    return emb_mean


# --- Spectrograms Viewer Class (unchanged from previous version) ---
class SpectrogramsViewer(tk.Toplevel):
    def __init__(self, master, selected_points, segment_length):
        super().__init__(master)
        self.title("Selected Spectrograms")
        self.geometry("1050x750")
        self.selected_points = sorted(list(set(selected_points)))
        self.segment_length = segment_length
        self.audio_cache = {}
        self.current_play_obj = None
        self.play_buttons = {}
        self.current_playing_sec = None
        self.current_playing_file = None
        self.animation_artists = {}
        self.is_playing = False
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.cmap_choice = tk.StringVar(value='inferno')
        self.freq_min = tk.IntVar(value=0)
        self.freq_max = tk.IntVar(value=5000)
        self.create_controls()
        self.create_scrollable_canvas()
        self.create_widgets()

    def create_scrollable_canvas(self):
        container = tb.Frame(self)
        container.pack(fill="both", expand=True)
        canvas = tk.Canvas(container)
        scrollbar = tb.Scrollbar(container, orient="vertical", command=canvas.yview)
        self.scroll_frame = tb.Frame(canvas)
        self.scroll_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def create_controls(self):
        control_frame = tb.Frame(self)
        control_frame.pack(fill="x", pady=5)
        cmap_label = tk.Label(control_frame, text="Colormap:")
        cmap_label.pack(side="left", padx=5)
        cmap_menu = tk.OptionMenu(
            control_frame, self.cmap_choice, 'inferno', 'viridis', 'plasma',
            'magma', 'cividis', 'gray', command=lambda _: self.update_spectrograms()
        )
        cmap_menu.pack(side="left", padx=5)
        tk.Label(control_frame, text="Min Freq").pack(side="left", padx=5)
        tk.Scale(control_frame, from_=0, to=20000, variable=self.freq_min,
                 orient='horizontal', length=150, command=lambda _: self.update_spectrograms()).pack(side="left")
        tk.Label(control_frame, text="Max Freq").pack(side="left", padx=5)
        tk.Scale(control_frame, from_=0, to=20000, variable=self.freq_max,
                 orient='horizontal', length=150, command=lambda _: self.update_spectrograms()).pack(side="left")

    def create_widgets(self):
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()
        self.play_buttons.clear()
        self.animation_artists.clear()

        cols = 2
        for idx, (sec, file_path) in enumerate(self.selected_points):
            try:
                frame = tb.Frame(self.scroll_frame, relief="ridge", borderwidth=2)
                frame.grid(row=idx // cols, column=idx % cols, padx=10, pady=10, sticky="nsew")

                if file_path not in self.audio_cache:
                    self.audio_cache[file_path] = sf.read(file_path)

                audio_data, sample_rate = self.audio_cache[file_path]

                start_sample = int(sec * sample_rate)
                end_sample = start_sample + int(self.segment_length * sample_rate)
                segment = audio_data[start_sample:end_sample]

                if len(segment) < 2:
                    print(f"Skipping spectrogram for {os.path.basename(file_path)} at {sec:.1f}s: segment too short.")
                    continue

                nperseg_val = 512
                if len(segment) < nperseg_val:
                    nperseg_val = len(segment)

                noverlap_val = nperseg_val // 2

                fig, ax = plt.subplots(figsize=(5, 3))
                f, t, Sxx = spectrogram(segment, fs=sample_rate, nperseg=nperseg_val, noverlap=noverlap_val)

                f_min = self.freq_min.get()
                f_max = self.freq_max.get()
                mask = (f >= f_min) & (f <= f_max)
                f, Sxx = f[mask], Sxx[mask, :]

                ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10),
                              shading='auto', cmap=self.cmap_choice.get())
                ax.set_title(f"{os.path.basename(file_path)}: {sec:.1f}s")
                ax.set_xlim(0, self.segment_length)
                ax.set_ylim(f_min, f_max)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Frequency (Hz)")
                plt.tight_layout()
                canvas_fig = FigureCanvasTkAgg(fig, master=frame)
                canvas_fig.draw()
                canvas_fig.get_tk_widget().pack()

                play_frame = tb.Frame(frame)
                play_frame.pack(fill="x", pady=5)
                play_btn = tb.Button(play_frame, text="▶", width=50,
                                     command=lambda s=sec, f=file_path, c=canvas_fig, a=ax: self.toggle_play_pause(s, f,
                                                                                                                   c,
                                                                                                                   a))
                play_btn.pack(side="left", padx=5)
                self.play_buttons[(sec, file_path)] = (play_btn, canvas_fig, ax)
            except Exception as e:
                print(f"Error creating spectrogram for {os.path.basename(file_path)} at {sec:.1f}s: {e}")

    def update_spectrograms(self):
        self.create_widgets()

    def toggle_play_pause(self, sec, file_path, canvas_fig, ax):
        if self.is_playing:
            if self.current_playing_sec == sec and self.current_playing_file == file_path:
                self.pause_audio()
            else:
                self.stop_audio()
                self.play_audio_segment(sec, file_path, canvas_fig, ax)
        else:
            self.play_audio_segment(sec, file_path, canvas_fig, ax)

    def play_audio_segment(self, sec, file_path, canvas_fig, ax):
        self.is_playing = True
        self.current_playing_sec = sec
        self.current_playing_file = file_path
        self.stop_event.clear()
        self.pause_event.clear()
        play_btn, _, _ = self.play_buttons[(sec, file_path)]
        play_btn.config(text="||")

        self.animation_artists[(sec, file_path)] = ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
        canvas_fig.draw_idle()

        # def play_thread():
        #     try:
        #         audio_data, sample_rate = self.audio_cache[file_path]
        #         start_frame = int(sec * sample_rate)
        #         end_frame = start_frame + int(self.segment_length * sample_rate)
        #         segment = audio_data[start_frame:end_frame]
        #
        #         audio_bytes = (segment * 32767).astype(np.int16).tobytes()
        #         play_obj = sa.play_buffer(audio_bytes, 1, 2, sample_rate)
        #
        #         start_time = time.time()
        #
        #         while play_obj.is_playing():
        #             if self.stop_event.is_set() or self.pause_event.is_set():
        #                 play_obj.stop()
        #                 break
        #             elapsed_time = time.time() - start_time
        #             if elapsed_time <= self.segment_length:
        #                 try:
        #                     self.after(10, lambda: self.update_animation(elapsed_time, sec, file_path))
        #                 except RuntimeError:
        #                     break
        #             time.sleep(0.01)
        #     except Exception as e:
        #         print(f"Error during playback: {e}")
        #     finally:
        #         self.after(0, lambda: self.reset_playback_state(sec, file_path))
        #
        # threading.Thread(target=play_thread, daemon=True).start()


        def play_thread():
            try:
                audio_data, sample_rate = self.audio_cache[file_path]

                # Extract segment
                start_frame = int(sec * sample_rate)
                end_frame = start_frame + int(self.segment_length * sample_rate)
                segment = audio_data[start_frame:end_frame]

                # If audio is stereo, take only first channel
                if segment.ndim > 1:
                    segment = segment[:, 0]

                start_time = time.time()

                # Playback in a separate thread
                def playback():
                    sd.play(segment, samplerate=sample_rate)
                    sd.wait()  # Wait until playback is finished

                threading.Thread(target=playback, daemon=True).start()

                # Animation loop while audio is playing
                while True:
                    elapsed_time = time.time() - start_time

                    # Stop if flagged
                    if self.stop_event.is_set() or self.pause_event.is_set() or elapsed_time >= self.segment_length:
                        sd.stop()
                        break

                    try:
                        self.after(10, lambda: self.update_animation(elapsed_time, sec, file_path))
                    except RuntimeError:
                        break

                    time.sleep(0.01)

            except Exception as e:
                print(f"Error during playback: {e}")
            finally:
                self.after(0, lambda: self.reset_playback_state(sec, file_path))

        # Start the playback thread
        threading.Thread(target=play_thread, daemon=True).start()

    def update_animation(self, current_time, sec, file_path):
        if (sec, file_path) in self.animation_artists and self.animation_artists[(sec, file_path)]:
            line = self.animation_artists[(sec, file_path)]
            line.set_xdata([current_time, current_time])
            self.play_buttons[(sec, file_path)][1].draw_idle()

    def pause_audio(self):
        if self.is_playing:
            self.pause_event.set()
            self.is_playing = False
            self.update_all_buttons()

    def stop_audio(self):
        if self.is_playing:
            self.stop_event.set()
            self.is_playing = False
            self.update_all_buttons()

    def reset_playback_state(self, sec, file_path):
        self.is_playing = False
        self.current_playing_sec = None
        self.current_playing_file = None
        play_btn, canvas_fig, _ = self.play_buttons[(sec, file_path)]
        play_btn.config(text="▶")
        if (sec, file_path) in self.animation_artists and self.animation_artists[(sec, file_path)]:
            self.animation_artists[(sec, file_path)].remove()
            self.animation_artists[(sec, file_path)] = None
            canvas_fig.draw_idle()

    def update_all_buttons(self):
        for key in self.play_buttons:
            btn, _, _ = self.play_buttons[key]
            btn.config(text="▶")

    def destroy(self):
        plt.close('all')
        super().destroy()








# --- VisualizationWindow Class (unchanged from previous version) ---
class VisualizationWindow(tk.Toplevel):
    def __init__(self, master, reduced_df, labels, feature_type, n_clusters_or_labels, segment_length, c_method, score,
                 file_paths, has_labels):
        super().__init__(master)
        self.score = score
        self.c_method = c_method
        self.file_paths = file_paths
        self.has_labels = has_labels
        self.title(f"{feature_type} Visualization")
        self.geometry("800x650")
        self.master_app = master
        self.reduced_df = reduced_df
        self.labels = labels
        self.feature_type = feature_type
        self.n_clusters_or_labels = n_clusters_or_labels
        self.segment_length = segment_length
        self.selected_markers = []

        # Main container
        main_frame = tb.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Add export button
        btn_frame = tb.Frame(main_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)

        self.export_btn = tb.Button(
            btn_frame,
            text="Export Labeled Data",
            bootstyle="success",
            command=self.export_labeled_data
        )
        self.export_btn.pack(side=tk.LEFT, padx=5)

        # Plot area
        self.fig, self.ax = plt.subplots(figsize=(7, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initialize cursor and lasso selector
        self.cursor = None
        self.lasso_selector = None
        self.plot_data()

    def plot_data(self):
        self.ax.clear()

        unique_files = self.reduced_df['file_name'].unique()
        file_cmap = plt.get_cmap('Dark2', len(unique_files))
        file_color_map = {file: file_cmap(i) for i, file in enumerate(unique_files)}

        unique_labels = np.unique(self.labels)
        cmap_labels = plt.get_cmap('viridis', len(unique_labels))

        self.scatter = self.ax.scatter(
            self.reduced_df['x'], self.reduced_df['y'],
            c=[cmap_labels(label) for label in self.labels],
            edgecolors=[file_color_map[f] for f in self.reduced_df['file_name']],
            linewidths=1.5,
            picker=True, s=50
        )

        if self.has_labels:
            self.ax.set_title(
                # f"{self.feature_type} Scatter Plot ({self.c_method})\n(N_Labels: {self.n_clusters_or_labels})")
                f"{self.c_method} Plot\n(N_Labels: {self.n_clusters_or_labels})")
        else:
            self.ax.set_title(
                # f"{self.feature_type} Scatter Plot({self.c_method})\n(N_Clusters: {self.n_clusters_or_labels}, Silhouette Score: {self.score:.2f})")
                f"{self.c_method} Plot\n(N_Clusters: {self.n_clusters_or_labels}, Silhouette Score: {self.score:.2f})")

        self.ax.set_xlabel("Dimension 1")
        self.ax.set_ylabel("Dimension 2")

        file_legend_handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=file_color_map[file],
                       markeredgecolor=file_color_map[file], markersize=10, label=os.path.basename(file))
            for file in unique_files
        ]

        label_title = "Labels" if self.has_labels else "Clusters"
        legend_handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap_labels(i),
                       markersize=10, label=f'{label_title} {int(i)}')
            for i in unique_labels if i != -1
        ]

        if -1 in unique_labels:
            legend_handles.append(
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, label='Noise'))

        file_legend = self.ax.legend(handles=file_legend_handles, title="Files", loc='upper left')
        self.ax.add_artist(file_legend)
        label_legend = self.ax.legend(handles=legend_handles, title=label_title, loc='lower right')

        seconds = self.reduced_df['second'].tolist()
        labels = self.labels.tolist()
        file_names = self.reduced_df['file_name'].tolist()

        # Initialize cursor with proper hover behavior
        if self.cursor:
            self.cursor.remove()

        self.cursor = mplcursors.cursor(self.scatter, hover=True, multiple=False)
        self.cursor.connect("add", lambda sel: self.on_hover(sel, seconds, labels, file_names))
        self.cursor.connect("remove", lambda sel: self.canvas.draw_idle())

        # Connect click event
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.draw()

    def on_hover(self, sel, seconds, labels, file_names):
        idx = sel.index
        sel.annotation.set_text(
            f"File: {os.path.basename(file_names[idx])}\nSecond: {seconds[idx]:.1f}\nLabel: {int(labels[idx])}"
        )
        # Set a timeout to remove the annotation if mouse leaves
        sel.annotation.set_animated(True)
        self.canvas.draw_idle()

    def on_click(self, event):
        if self.reduced_df is None or event.inaxes != self.ax:
            return

        distances = np.hypot(
            self.reduced_df['x'] - event.xdata, self.reduced_df['y'] - event.ydata
        )
        min_idx = np.argmin(distances)

        if distances[min_idx] > 0.05 * (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]):
            return

        sec = self.reduced_df.iloc[min_idx]['second']
        file_name = self.reduced_df.iloc[min_idx]['file_name']

        if (sec, file_name) not in self.master_app.selected_seconds:
            self.master_app.selected_seconds.append((sec, file_name))
            self.master_app.selected_listbox.insert(tk.END, f"{os.path.basename(file_name)}: {sec:.1f}s")
            marker = self.ax.scatter(
                self.reduced_df.iloc[min_idx]['x'], self.reduced_df.iloc[min_idx]['y'],
                s=200, facecolors='none', edgecolors='black', linewidths=2
            )
            self.selected_markers.append(marker)
            self.canvas.draw()

    def export_labeled_data(self):
        """Export the current data with cluster labels as a CSV file."""
        try:
            # Create a copy of the reduced dataframe
            export_df = self.reduced_df.copy()

            # Add the cluster labels
            export_df['label'] = self.labels

            # Get the original feature columns from cache
            feature_cols = []
            for file_path in self.file_paths:
                cache_key = (file_path, self.feature_type, self.segment_length)
                if cache_key in self.master_app.cached_features:
                    features = self.master_app.cached_features[cache_key]
                    feature_cols = [col for col in features.columns if col not in ['second', 'file_name']]
                    break

            # If we have the original features, merge them
            if feature_cols:
                # Combine all original features
                all_features = []
                for file_path in self.file_paths:
                    cache_key = (file_path, self.feature_type, self.segment_length)
                    if cache_key in self.master_app.cached_features:
                        features = self.master_app.cached_features[cache_key].copy()
                        features['file_name'] = file_path
                        all_features.append(features)

                if all_features:
                    combined_features = pd.concat(all_features, ignore_index=True)
                    export_df = pd.merge(
                        export_df,
                        combined_features,
                        on=['second', 'file_name'],
                        # on=[],
                        how='right'
                    )

            # Save to file
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                title="Save labeled data as"
            )

            if file_path:
                export_df.to_csv(file_path, index=False)
                messagebox.showinfo("Success", f"Data exported successfully to {file_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to export data: {str(e)}")

    def destroy(self):
        # Clean up matplotlib objects
        if self.cursor:
            self.cursor.remove()
        if self.lasso_selector:
            self.lasso_selector.disconnect_events()
        plt.close(self.fig)
        super().destroy()


# --- New Machine Learning Classification Window ---
class MLClassifierWindow(tk.Toplevel):
    def __init__(self, master, reduced_df):
        super().__init__(master)
        self.title("Machine Learning Classification")
        self.geometry("600x500")
        self.master = master
        self.reduced_df = reduced_df

        self.model_var = tk.StringVar(value="Random Forest")
        self.split_var = tk.IntVar(value=80)

        self.create_widgets()

    def create_widgets(self):
        # Frame for controls
        control_frame = tb.Frame(self)
        control_frame.pack(padx=10, pady=10, fill='x')

        # Model Selection
        tb.Label(control_frame, text="Select Model", font=("Segoe UI", 10, "bold")).pack(anchor='w', pady=(0, 5))
        model_options = tb.Frame(control_frame)
        model_options.pack(fill='x')
        tb.Radiobutton(model_options, text="Random Forest", variable=self.model_var, value="Random Forest").pack(
            side='left', padx=5)
        tb.Radiobutton(model_options, text="Decision Tree", variable=self.model_var, value="Decision Tree").pack(
            side='left', padx=5)
        tb.Radiobutton(model_options, text="Gradient Boosting", variable=self.model_var,
                       value="Gradient Boosting").pack(side='left', padx=5)
        tb.Radiobutton(model_options, text="Linear SVM", variable=self.model_var, value="Linear SVM").pack(side='left',
                                                                                                           padx=5)
        tb.Radiobutton(model_options, text="Voting Classifier", variable=self.model_var,
                       value="Voting Classifier").pack(side='left', padx=5)

        # Data Split
        tb.Label(control_frame, text="Training Data Split (%)", font=("Segoe UI", 10, "bold")).pack(anchor='w',
                                                                                                    pady=(10, 5))
        split_frame = tb.Frame(control_frame)
        split_frame.pack(fill='x')
        tk.Scale(split_frame, from_=50, to=90, variable=self.split_var, orient='horizontal', resolution=5,
                 length=400).pack(side='left', fill='x', expand=True)
        tb.Label(split_frame, textvariable=self.split_var).pack(side='left', padx=5)

        # Run Button
        self.run_btn = tb.Button(control_frame, text="Run Classification", bootstyle="primary",
                                 command=self.run_classification_in_thread)
        self.run_btn.pack(fill='x', pady=(15, 5))

        # Frame for Results Display (horizontal layout using grid)
        self.results_frame = tb.Frame(self)
        self.results_frame.pack(padx=10, pady=10, fill='both', expand=True)

        # Configure the grid to have two columns with specified weights
        self.results_frame.grid_columnconfigure(0, weight=90)
        self.results_frame.grid_columnconfigure(1, weight=10)

        # Left side: Heatmap Canvas (65% width)
        self.heatmap_frame = tb.Frame(self.results_frame)
        self.heatmap_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
        # We will create the canvas inside display_results
        self.cm_canvas = None

        # Right side: Text Results (30% width)
        self.results_text_frame = tb.Frame(self.results_frame)
        self.results_text_frame.grid(row=0, column=1, sticky='nsew', padx=(5, 0))
        self.results_label = tb.Label(self.results_text_frame, text="Results will be displayed here.",
                                      font=("Segoe UI", 15, "bold"))
        self.results_label.pack(anchor='w', pady=(0, 5))
        self.results_text = scrolledtext.ScrolledText(self.results_text_frame, wrap=tk.WORD, height=15,
                                                      state='disabled')
        self.results_text.pack(fill='both', expand=True)

    def run_classification_in_thread(self):
        self.run_btn.config(state="disabled")
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Running classification, please wait...")
        self.results_text.config(state='disabled')
        self.results_label.config(text="Results:")

        threading.Thread(target=self.run_classification, daemon=True).start()

    def run_classification(self):
        """
        Performs machine learning classification on the reduced audio data.
        """
        try:
            # Check if labeled data is available
            if 'label' not in self.reduced_df.columns:
                messagebox.showerror("Error", "Cannot perform classification: features are not labeled.")
                return

            # Prepare data
            feature_cols = [col for col in self.reduced_df.columns if col not in ['second', 'file_name', 'label']]
            X = self.reduced_df[feature_cols]
            y = self.reduced_df['label'].values.astype(int)

            if len(np.unique(y)) < 2:
                messagebox.showerror("Error", "Cannot perform classification: less than 2 unique labels.")
                return

            # Split data
            split_ratio = self.split_var.get() / 100.0
            test_size = 1 - split_ratio
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

            # Select and train model
            model_choice = self.model_var.get()
            model = None
            if model_choice == "Random Forest":
                model = RandomForestClassifier(random_state=42)
            elif model_choice == "Decision Tree":
                model = DecisionTreeClassifier(random_state=42)
            elif model_choice == "Gradient Boosting":
                model = GradientBoostingClassifier(random_state=42)
            elif model_choice == "Linear SVM":
                model = SVC(kernel='linear', random_state=42)
            elif model_choice == "Voting Classifier":
                # Create a list of the base estimators
                estimators = [
                    ('rf', RandomForestClassifier(random_state=42)),
                    ('dt', DecisionTreeClassifier(random_state=42)),
                    ('svm', SVC(kernel='linear', random_state=42, probability=True))
                ]
                model = VotingClassifier(estimators=estimators, voting='soft')  # Use soft voting for better performance

            if model:
                # Train the model
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred)

                # --- COMPUTE CONFUSION MATRIX ---
                cm = confusion_matrix(y_test, y_pred)

                # Display results
                self.after(0, lambda: self.display_results(accuracy, report, cm))

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        finally:
            self.after(0, lambda: self.run_btn.config(state="normal"))

    def display_results(self, accuracy, report, cm):
        """
        Displays the classification results including accuracy, confusion matrix heatmap, and classification report.
        """
        # Create the confusion matrix heatmap
        self.cm_figure = plt.Figure(figsize=(5, 4), dpi=100)
        ax = self.cm_figure.add_subplot(111)

        # Get unique labels for the heatmap
        labels = sorted(self.reduced_df['label'].unique())
        label_map = {label: i for i, label in enumerate(labels)}

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=labels, yticklabels=labels)
        ax.set_title('Confusion Matrix', fontsize=12)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

        # Save the plot to a buffer
        buf = io.BytesIO()
        self.cm_figure.tight_layout()
        self.cm_figure.savefig(buf, format='png')
        plt.close(self.cm_figure)  # Close the figure to free memory

        # Load the image into Tkinter
        buf.seek(0)
        img = Image.open(buf)
        self.cm_photo = ImageTk.PhotoImage(img)

        # Display the image in a new canvas or label
        # Create the canvas inside the new heatmap_frame
        if self.cm_canvas is None:
            self.cm_canvas = tk.Canvas(self.heatmap_frame, width=500, height=400)
            self.cm_canvas.pack(fill='both', expand=True, pady=10)

        self.cm_canvas.delete("all")
        self.cm_canvas.create_image(0, 0, anchor=tk.NW, image=self.cm_photo)

        # Update the text widget with accuracy and report
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)

        result_text = f"Model: {self.model_var.get()}\n"
        result_text += f"Training/Testing Split: {self.split_var.get()}/{100 - self.split_var.get()}%\n\n"
        result_text += f"Accuracy: {accuracy:.4f}\n\n"
        result_text += "Classification Report:\n"
        result_text += "----------------------\n"
        result_text += report

        self.results_text.insert(tk.END, result_text)
        self.results_text.config(state='disabled')


# --- Main Application Class ---
class AudioVisualizerApp(tb.Window):
    def __init__(self):
        super().__init__(themename="superhero")
        self.title("AI Eco-Sound Tutor")
        self.geometry("1100x750")

        self.file_paths = []
        self.file_type = None
        self.reduced_df = None
        self.labels = None
        self.has_labels = False
        self.selected_seconds = []
        self.cached_features = {}
        self.selected_markers = []

        self.lasso_mode = False
        self.lasso_selector = None

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Main frames
        self.ctrl_frame = tb.Frame(self)
        self.ctrl_frame.grid(row=0, column=0, sticky="ns", padx=10, pady=10)
        self.display_frame = tb.Frame(self)
        self.display_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.create_controls()
        self.create_display()
        self.update_ui_state()

    def create_controls(self):
        # NEW: Create a Canvas and Scrollbar to make the control frame scrollable
        canvas = tk.Canvas(self.ctrl_frame, highlightthickness=0)
        scrollbar = tb.Scrollbar(self.ctrl_frame, orient="vertical", command=canvas.yview)

        self.scrollable_frame = tb.Frame(canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # All widgets are now parented to self.scrollable_frame
        # Row layout is now relative to this frame.
        row_idx = 0
        self.upload_btn = tb.Button(self.scrollable_frame, text="Upload .wav or .csv", bootstyle="info-outline",
                                    command=self.load_file)
        self.upload_btn.grid(row=row_idx, column=0, sticky="ew", pady=(0, 5))
        row_idx += 1

        self.file_label = tb.Label(self.scrollable_frame, text="No file uploaded", wraplength=200, justify="left")
        self.file_label.grid(row=row_idx, column=0, sticky="w", pady=(0, 15))
        row_idx += 1

        tb.Label(self.scrollable_frame, text="Feature Type", font=("Segoe UI", 10, "bold")).grid(row=row_idx, column=0,
                                                                                                 sticky="w")
        # info_btn = tk.Button(self.scrollable_frame, text="?", width=2, command=lambda: ToolTip.show_info(
        #     "Feature Type(s)",
        #     text="",
        #     link="https://en.wikipedia.org/wiki/Mel-frequency_cepstrum"
        # ))
        # info_btn.grid(row=row_idx, column=1)

        row_idx += 1
        self.feature_type_var = tk.StringVar(value="MFCC")
        self.ft_frame = tb.Frame(self.scrollable_frame)
        self.ft_frame.grid(row=row_idx, column=0, sticky="w", pady=(0, 15))
        self.mfcc_rb = tb.Radiobutton(self.ft_frame, text="MFCC", variable=self.feature_type_var, value="MFCC")
        self.mfcc_rb.grid(row=0, column=0, padx=5)
        self.openl3_rb = tb.Radiobutton(self.ft_frame, text="OpenL3", variable=self.feature_type_var, value="OpenL3")
        self.openl3_rb.grid(row=0, column=1, padx=5)
        row_idx += 1


        tb.Label(self.scrollable_frame, text="Segment Length (s)", font=("Segoe UI", 10, "bold")).grid(row=row_idx,
                                                                                                       column=0,
                                                                                                       sticky="w")


        row_idx += 1
        self.segment_length_var = tk.DoubleVar(value=1.0)
        self.slider_frame = tb.Frame(self.scrollable_frame)
        self.slider_frame.grid(row=row_idx, column=0, sticky="ew", pady=(0, 15))
        self.segment_length_slider = tk.Scale(
            self.slider_frame, from_=0.1, to=10.0, variable=self.segment_length_var, orient='horizontal',
            resolution=0.1, length=250,
            command=self.on_segment_length_change
        )
        self.segment_length_slider.pack(side="left", fill="x", expand=True)
        self.segment_length_label = tb.Label(self.slider_frame, textvariable=self.segment_length_var)
        self.segment_length_label.pack(side="left", padx=5)

        # info_btn2 = tk.Button(self.scrollable_frame, text="?", width=2, command=lambda: ToolTip.show_info(
        #     title="Segment Length",
        #     text="",
        #     link="https://en.wikipedia.org/wiki/Mel-frequency_cepstrum"
        # ))
        # info_btn2.grid(row=row_idx, column=2)

        row_idx += 1

        tb.Label(self.scrollable_frame, text="Dimensionality Reduction", font=("Segoe UI", 10, "bold")).grid(
            row=row_idx, column=0, sticky="w")

        # info_btn3 = tk.Button(self.scrollable_frame, text="?", width=2, command=lambda: ToolTip.show_info(
        #     "Dimensionality Reduction",
        #     "",
        #     "https://en.wikipedia.org/wiki/Mel-frequency_cepstrum"
        # ))
        # info_btn3.grid(row=row_idx, column=1)

        row_idx += 1
        self.method_var = tk.StringVar(value="PCA")
        method_frame = tb.Frame(self.scrollable_frame)
        method_frame.grid(row=row_idx, column=0, sticky="w", pady=(0, 15))
        for i, m in enumerate(["PCA", "t-SNE", "UMAP"]):
            tb.Radiobutton(method_frame, text=m, variable=self.method_var, value=m).grid(row=0, column=i, padx=5)
        row_idx += 1

        tb.Label(self.scrollable_frame, text="Clustering Algorithm", font=("Segoe UI", 10, "bold")).grid(row=row_idx,
                                                                                                         column=0,
                                                                                                         sticky="w")
        # info_btn4 = tk.Button(self.scrollable_frame, text="?", width=2, command=lambda: ToolTip.show_info(
        #     "Clustering",
        #     "",
        #     "https://en.wikipedia.org/wiki/Mel-frequency_cepstrum"
        # ))
        # info_btn4.grid(row=row_idx, column=1)

        row_idx += 1
        self.cluster_var = tk.StringVar(value="None")
        self.cluster_frame = tb.Frame(self.scrollable_frame)
        self.cluster_frame.grid(row=row_idx, column=0, sticky="w", pady=(0, 15))
        for i, c in enumerate(["None", "KMeans", "GMM", "HDBSCAN"]):
            tb.Radiobutton(self.cluster_frame, text=c, variable=self.cluster_var, value=c).grid(row=0, column=i, padx=5)
        row_idx += 1

        tb.Label(self.scrollable_frame, text="Number of Clusters", font=("Segoe UI", 10, "bold")).grid(row=row_idx,
                                                                                                       column=0,
                                                                                                       sticky="w")
        row_idx += 1
        self.n_clusters_var = tk.IntVar(value=2)
        self.cluster_slider_frame = tb.Frame(self.scrollable_frame)
        self.cluster_slider_frame.grid(row=row_idx, column=0, sticky="ew", pady=(0, 15))
        self.cluster_slider = tk.Scale(
            self.cluster_slider_frame, from_=2, to=10, orient=tk.HORIZONTAL, resolution=1,
            variable=self.n_clusters_var, command=self.update_cluster_label
        )
        self.cluster_slider.pack(side="left", fill="x", expand=True)
        self.cluster_value_label = tb.Label(self.cluster_slider_frame, text="2")
        self.cluster_value_label.pack(side="left", padx=5)
        self.update_slider_state()
        row_idx += 1

        visualization_buttons_frame = tb.Frame(self.scrollable_frame)
        visualization_buttons_frame.grid(row=row_idx, column=0, sticky="ew", pady=(10, 5))
        self.visualize_btn = tb.Button(visualization_buttons_frame, text="VISUALIZE", bootstyle="success-outline",
                                       command=lambda: self.start_visualization(in_new_window=False))
        self.visualize_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.visualize_new_btn = tb.Button(visualization_buttons_frame, text="Visualize (New Window)",
                                           bootstyle="success-outline",
                                           command=lambda: self.start_visualization(in_new_window=True))
        self.visualize_new_btn.pack(side="right", fill="x", expand=True, padx=(5, 0))
        row_idx += 1

        self.progress_label = tb.Label(self.scrollable_frame, text="")
        self.progress_label.grid(row=row_idx, column=0, sticky="w", pady=(10, 5))
        row_idx += 1
        self.progress_bar = tb.Progressbar(self.scrollable_frame, mode="determinate")
        self.progress_bar.grid(row=row_idx, column=0, sticky="ew", pady=(5, 10))
        row_idx += 1

        self.ml_btn = tb.Button(self.scrollable_frame, text="Run ML Classification", bootstyle="primary-outline",
                                command=self.open_ml_window)
        self.ml_btn.grid(row=row_idx, column=0, sticky="ew", pady=(15, 10))
        row_idx += 1

        tb.Label(self.scrollable_frame, text="Selection Tools", font=("Segoe UI", 10, "bold")).grid(row=row_idx,
                                                                                                    column=0,
                                                                                                    sticky="w")
        row_idx += 1
        selection_frame = tb.Frame(self.scrollable_frame)
        selection_frame.grid(row=row_idx, column=0, sticky="ew", pady=(0, 10))
        self.lasso_btn = tb.Button(selection_frame, text="Lasso Select", bootstyle="warning-outline",
                                   command=self.toggle_lasso_mode)
        self.lasso_btn.pack(side="left", fill="x", expand=True)
        row_idx += 1

        tb.Label(self.scrollable_frame, text="Selected Points", font=("Segoe UI", 10, "bold")).grid(row=row_idx,
                                                                                                    column=0,
                                                                                                    sticky="w")
        row_idx += 1
        self.selected_listbox = tk.Listbox(self.scrollable_frame, height=8)
        self.selected_listbox.grid(row=row_idx, column=0, sticky="ew", pady=(0, 10))
        row_idx += 1
        self.reset_btn = tb.Button(self.scrollable_frame, text="Reset Selected", bootstyle="danger-outline",
                                   command=self.reset_selected_points)
        self.reset_btn.grid(row=row_idx, column=0, sticky="ew", pady=(5, 5))
        row_idx += 1
        self.view_btn = tb.Button(self.scrollable_frame, text="View Selected Spectrograms", bootstyle="primary-outline",
                                  command=self.open_spectrogram_window)
        self.view_btn.grid(row=row_idx, column=0, sticky="ew")
        row_idx += 1

        self.cluster_var.trace_add("write", lambda *args: self.update_slider_state())

    def create_display(self):
        self.fig, self.ax = plt.subplots(figsize=(7, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.display_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax.set_title("Feature Scatter Plot")
        self.ax.set_xlabel("Dimension 1")
        self.ax.set_ylabel("Dimension 2")
        self.click_cid = self.canvas.mpl_connect("button_press_event", self.on_click)

    def open_ml_window(self):
        if self.reduced_df is None:
            messagebox.showinfo("Info", "Please visualize data with labels first.")
            return
        if not self.has_labels:
            messagebox.showinfo("Info", "ML classification requires labeled data (.csv file with a 'label' column).")
            return
        MLClassifierWindow(self, self.reduced_df)

    def toggle_lasso_mode(self):
        self.lasso_mode = not self.lasso_mode
        if self.lasso_mode:
            if self.reduced_df is None:
                messagebox.showinfo("Info", "Please visualize data before using the lasso tool.")
                self.lasso_mode = False
                return

            self.lasso_btn.config(bootstyle="warning")
            self.canvas.mpl_disconnect(self.click_cid)

            # Clear any existing lasso selector
            if hasattr(self, 'lasso_selector') and self.lasso_selector:
                self.lasso_selector.disconnect_events()

            def on_select(verts):
                try:
                    path = Path(verts)
                    points_xy = self.reduced_df[['x', 'y']].values
                    inside_points = path.contains_points(points_xy)

                    # Get all selected rows at once using the boolean array
                    selected_df = self.reduced_df[inside_points]

                    # Clear previous selections
                    self.reset_selected_points()

                    # Batch the marker creation with a single scatter plot call
                    if not selected_df.empty:
                        new_markers = self.ax.scatter(
                            selected_df['x'], selected_df['y'],
                            s=200, facecolors='none', edgecolors='black', linewidths=2
                        )
                        self.selected_markers.append(new_markers)

                    # Batch the listbox updates
                    selected_seconds = selected_df[['second', 'file_name']].apply(
                        lambda row: (row['second'], row['file_name']), axis=1
                    ).tolist()

                    for sec, file_name in selected_seconds:
                        if (sec, file_name) not in self.selected_seconds:
                            self.selected_seconds.append((sec, file_name))
                            self.selected_listbox.insert(tk.END, f"{os.path.basename(file_name)}: {sec:.1f}s")

                    self.canvas.draw()
                except Exception as e:
                    print(f"Error in lasso selection: {e}")

            self.lasso_selector = LassoSelector(self.ax, on_select)
            self.lasso_selector.set_active(True)
        else:
            self.lasso_btn.config(bootstyle="warning-outline")
            if hasattr(self, 'lasso_selector') and self.lasso_selector:
                self.lasso_selector.disconnect_events()
                self.lasso_selector = None
            self.click_cid = self.canvas.mpl_connect("button_press_event", self.on_click)

        self.canvas.draw()

        #     def on_select(verts):
        #         if verts is None:
        #             return
        #         path = Path(verts)
        #         points_xy = self.reduced_df[['x', 'y']].values
        #         inside_points = path.contains_points(points_xy)
        #
        #         for i, is_inside in enumerate(inside_points):
        #             if is_inside:
        #                 sec = self.reduced_df.iloc[i]['second']
        #                 file_name = self.reduced_df.iloc[i]['file_name']
        #                 if (sec, file_name) not in self.selected_seconds:
        #                     self.selected_seconds.append((sec, file_name))
        #                     self.selected_listbox.insert(tk.END, f"{os.path.basename(file_name)}: {sec:.1f}s")
        #                     marker = self.ax.scatter(
        #                         self.reduced_df.iloc[i]['x'], self.reduced_df.iloc[i]['y'],
        #                         s=200, facecolors='none', edgecolors='black', linewidths=2
        #                     )
        #                     self.selected_markers.append(marker)
        #         self.canvas.draw()
        #
        #     self.lasso_selector = LassoSelector(self.ax, on_select)
        # else:
        #     self.lasso_btn.config(bootstyle="warning-outline")
        #     if self.lasso_selector:
        #         self.lasso_selector.disconnect_events()
        #         self.lasso_selector = None
        #     self.click_cid = self.canvas.mpl_connect("button_press_event", self.on_click)
        #     self.canvas.draw()

    def on_segment_length_change(self, val):
        self.cached_features.clear()

    def update_slider_state(self, *args):
        disable_clustering = self.has_labels

        if disable_clustering:
            state = "disabled"
            self.cluster_value_label.config(text="N/A")
            self.cluster_var.set("None")
        else:
            state = "normal"
            is_hdbscan = self.cluster_var.get() == "HDBSCAN"
            is_none = self.cluster_var.get() == "None"
            if is_hdbscan or is_none:
                self.cluster_slider.configure(state="disabled")
            else:
                self.cluster_slider.configure(state="normal")

            if is_hdbscan:
                self.cluster_value_label.config(text="Auto")
            elif not is_none:
                self.update_cluster_label(self.n_clusters_var.get())
            else:
                self.cluster_value_label.config(text="N/A")

        for child in self.cluster_frame.winfo_children():
            child.configure(state=state)
        self.cluster_slider.configure(state=state)

    def update_cluster_label(self, value):
        self.cluster_value_label.config(text=str(int(value)))

    def update_ui_state(self):
        """Disables/enables UI elements based on loaded file type and labels."""
        is_wav = self.file_type == 'wav'

        audio_ctrl_state = 'normal' if is_wav else 'disabled'
        self.mfcc_rb.configure(state=audio_ctrl_state)
        self.openl3_rb.configure(state=audio_ctrl_state)
        self.segment_length_slider.configure(state=audio_ctrl_state)
        self.view_btn.configure(state=audio_ctrl_state)

        self.lasso_btn.configure(state='normal' if self.reduced_df is not None else 'disabled')

        vis_state = 'normal' if self.file_paths else 'disabled'
        self.visualize_btn.configure(state=vis_state)
        self.visualize_new_btn.configure(state=vis_state)
        self.ml_btn.configure(state='normal' if self.has_labels and self.reduced_df is not None else 'disabled')

        self.update_slider_state()

    def load_file(self):
        file_paths = filedialog.askopenfilenames(
            title="Select WAV or CSV files",
            filetypes=[("Supported files", "*.wav *.csv"), ("WAV files", "*.wav"), ("CSV files", "*.csv")]
        )
        if not file_paths:
            return

        first_file_ext = os.path.splitext(file_paths[0])[1].lower()
        if not all(os.path.splitext(fp)[1].lower() == first_file_ext for fp in file_paths):
            messagebox.showerror("Error", "Please select files of the same type (.wav or .csv).")
            self.file_paths = []
            self.file_type = None
            self.file_label.configure(text="No file uploaded")
            self.has_labels = False
            self.update_ui_state()
            return

        self.file_paths = list(file_paths)
        self.file_type = 'wav' if first_file_ext == '.wav' else 'csv'

        self.has_labels = False
        if self.file_type == 'csv':
            try:
                temp_df = pd.read_csv(self.file_paths[0])
                if 'label' in temp_df.columns:
                    self.has_labels = True
                    self.cluster_var.set("None")
                    temp_df['label'] = pd.to_numeric(temp_df['label'], errors='coerce')
                    temp_df.dropna(subset=['label'], inplace=True)
                    if temp_df['label'].nunique() < 2:
                        self.has_labels = False
                        messagebox.showinfo("Info", "Less than 2 unique labels found. Classification disabled.")
            except Exception as e:
                print(f"Error checking CSV file for labels: {e}")
                self.has_labels = False

        display_names = [os.path.basename(fp) for fp in self.file_paths]
        self.file_label.configure(text=f"Loaded: {', '.join(display_names)}")

        self.cached_features.clear()
        self.selected_seconds.clear()
        self.selected_listbox.delete(0, tk.END)
        self.reset_selected_points()
        self.update_ui_state()

    def start_visualization(self, in_new_window):
        if not self.file_paths:
            messagebox.showerror("Error", "Please load a file first.")
            return

        self.visualize_btn.config(state="disabled")
        self.visualize_new_btn.config(state="disabled")

        threading.Thread(target=self.visualize, args=(in_new_window,), daemon=True).start()

    def visualize(self, in_new_window):
        try:
            self.update_progress("Processing files...", 10)
            all_features = []

            for i, file_path in enumerate(self.file_paths):
                self.update_progress(f"Processing {os.path.basename(file_path)}...",
                                     10 + int(i / len(self.file_paths) * 30))

                cache_key = (file_path, self.feature_type_var.get(), self.segment_length_var.get())
                if cache_key in self.cached_features:
                    features = self.cached_features[cache_key]
                else:
                    if self.file_type == 'wav':
                        if self.feature_type_var.get() == "MFCC":
                            features = extract_mfcc(file_path, segment_length=self.segment_length_var.get())
                        else:
                            features = extract_openl3(file_path, segment_length=self.segment_length_var.get())
                    else:
                        features = pd.read_csv(file_path)
                        if 'second' not in features.columns:
                            features['second'] = [i * 1.0 for i in range(len(features))]
                    self.cached_features[cache_key] = features

                features['file_name'] = file_path
                all_features.append(features)

            combined_features = pd.concat(all_features, ignore_index=True)

            self.update_progress("Reducing dimensions...", 40)
            method = self.method_var.get()
            self.reduced_df = reduce_dim(combined_features, method)

            self.update_progress("Clustering data...", 70)
            labels = None
            s_score = 0.0
            n_clusters_or_labels = 0

            if self.has_labels:
                labels = self.reduced_df['label'].values.astype(int)
                n_clusters_or_labels = len(np.unique(labels))
                self.labels = labels
            else:
                cluster_algo = self.cluster_var.get()
                n_clusters = self.n_clusters_var.get()

                if cluster_algo == "None":
                    labels = np.zeros(len(self.reduced_df))
                else:
                    labels, s_score, n_clusters_or_labels = apply_clustering(self.reduced_df, cluster_algo, n_clusters)

                self.labels = labels
                n_clusters_or_labels = n_clusters_or_labels

            self.update_progress("Plotting...", 90)

            if in_new_window:
                VisualizationWindow(
                    self, self.reduced_df, labels, self.feature_type_var.get(), n_clusters_or_labels,
                    self.segment_length_var.get() if self.file_type == 'wav' else 0,
                    self.method_var.get(), score=s_score, file_paths=self.file_paths, has_labels=self.has_labels
                )
            else:
                self.plot_data(self.reduced_df, labels, self.feature_type_var.get(), n_clusters_or_labels,
                               self.segment_length_var.get() if self.file_type == 'wav' else 0,
                               self.method_var.get(), s_score, self.has_labels)

            self.update_progress("Done", 100)

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.update_progress("Error occurred", 0)
        finally:
            self.visualize_btn.config(state="normal")
            self.visualize_new_btn.config(state="normal")
            self.update_ui_state()

    def plot_data(self, reduced, labels, ft, n_clusters_or_labels, segment_len, method, score, has_labels):
        self.ax.clear()

        unique_files = reduced['file_name'].unique()
        file_cmap = plt.get_cmap('Dark2', len(unique_files))
        file_color_map = {file: file_cmap(i) for i, file in enumerate(unique_files)}

        unique_labels = np.unique(labels)
        cmap_labels = plt.get_cmap('viridis', len(unique_labels))

        scatter = self.ax.scatter(
            reduced['x'], reduced['y'],
            c=[cmap_labels(label) for label in labels],
            edgecolors=[file_color_map[f] for f in reduced['file_name']],
            linewidths=1.5,
            picker=True, s=50
        )

        if has_labels:
            # self.ax.set_title(f"{ft} Scatter Plot ({method})\n(N_Labels: {n_clusters_or_labels})")
            self.ax.set_title(f"{method} Plot \n (N_Labels: {n_clusters_or_labels})")
        else:
            self.ax.set_title(
                # f"{ft} Scatter Plot ({method})\n(N_Clusters: {n_clusters_or_labels}, Silhouette Score: {score:.2f})")
                f"{method} Plot \n (N_Clusters: {n_clusters_or_labels}, Silhouette Score: {score:.2f})")

        self.ax.set_xlabel("Dimension 1")
        self.ax.set_ylabel("Dimension 2")

        file_legend_handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=file_color_map[file],
                       markeredgecolor=file_color_map[file], markersize=10, label=os.path.basename(file))
            for file in unique_files
        ]

        label_title = "Labels" if has_labels else "Clusters"
        legend_handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap_labels(i),
                       markersize=10, label=f'{label_title} {int(i)}')
            for i in unique_labels if i != -1
        ]

        if -1 in unique_labels:
            legend_handles.append(
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, label='Noise'))

        file_legend = self.ax.legend(handles=file_legend_handles, title="Files", loc='lower left')
        self.ax.add_artist(file_legend)
        label_legend = self.ax.legend(handles=legend_handles, title=label_title, loc='upper right')

        seconds = reduced['second'].tolist()
        labels = labels.tolist()
        file_names = reduced['file_name'].tolist()

        cursor = mplcursors.cursor(scatter, hover=True)

        @cursor.connect("add")
        def on_hover(sel):
            idx = sel.index
            sel.annotation.set_text(
                f"File: {os.path.basename(file_names[idx])}\nPoint: {seconds[idx]:.1f}\nLabel: {int(labels[idx])}"
            )

        self.canvas.draw()

    def on_click(self, event):
        if self.reduced_df is None or event.inaxes != self.ax:
            return

        distances = np.hypot(
            self.reduced_df['x'] - event.xdata, self.reduced_df['y'] - event.ydata
        )
        min_idx = np.argmin(distances)

        if distances[min_idx] > 0.05 * (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]):
            return

        sec = self.reduced_df.iloc[min_idx]['second']
        file_name = self.reduced_df.iloc[min_idx]['file_name']

        if (sec, file_name) not in self.selected_seconds:
            self.selected_seconds.append((sec, file_name))
            self.selected_listbox.insert(tk.END, f"{os.path.basename(file_name)}: {sec:.1f}s")
            marker = self.ax.scatter(
                self.reduced_df.iloc[min_idx]['x'], self.reduced_df.iloc[min_idx]['y'],
                s=200, facecolors='none', edgecolors='black', linewidths=2
            )
            self.selected_markers.append(marker)
            self.canvas.draw()

    def update_progress(self, text, value):
        self.progress_label.configure(text=text)
        self.progress_bar['value'] = value
        self.update_idletasks()

    def reset_selected_points(self):
        self.selected_seconds.clear()
        self.selected_listbox.delete(0, tk.END)
        for marker in self.selected_markers:
            try:
                marker.remove()
            except NotImplementedError:
                marker.set_visible(False)
        self.selected_markers.clear()
        if self.canvas:
            self.canvas.draw()

    def open_spectrogram_window(self):
        if self.file_type != 'wav':
            messagebox.showinfo("Info", "Spectrograms are only available for WAV files.")
            return

        if not self.selected_seconds:
            messagebox.showinfo("Info", "No points selected.")
            return

        SpectrogramsViewer(self, self.selected_seconds, self.segment_length_var.get())


if __name__ == "__main__":
    app = AudioVisualizerApp()
    app.mainloop()
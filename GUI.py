import tkinter as tk
#from tkinter import scrolledtext
from tkinter import ttk
from YT_API import yt_extract
from yt_detection import youtube_detection


def scan_comment_section():
    url = field_url.get()
    vect_name = vect_name_var.get()
    model_name = model_name_var.get()
    lemmatization = lemmatization_var.get()
    try:
        limit = int(field_limit.get())
    except ValueError:
        limit = 10
        print(f"No limit / Wrong input -> limit has been set to {limit}")
    yt_comments, title = yt_extract(url = url, limit = limit)
    predicted = youtube_detection(yt_comments, vect_name= vect_name, model_name= model_name, lemmatization= lemmatization)
    yt_comments['IsToxic'] = predicted
    # Czyszczenie tabeli
    for i in table.get_children():
        table.delete(i)
    # Dodawanie nowych wierszy do tabeli
    for _, wiersz in yt_comments.iterrows():
        IsToxic = 'TOKSYCZNY' if wiersz['IsToxic'] else 'Bezpieczny'
        table.insert("", 'end', values=(wiersz['Comment'], IsToxic))

def update_columns(event):
    """
    Function for keeping the right proportion for displayed columns in table
    """
    table_width = table.winfo_width()
    is_toxic_width = 80
    table.column("Comment", width=int(table_width-is_toxic_width))
    table.column("IsToxic", width=int(is_toxic_width))

# Tkinter start
window = tk.Tk()
window.title("Detektor mowy nienawiści na polskojęzycznym YouTube")

# Rozmiar okna
window_height = 800
window_width = 1200
window.geometry(f"{window_width}x{window_height}")

# URL
label_url = tk.Label(window, text="Link do filmu na portalu YouTube:")
label_url.pack(padx=10, pady=(10,0))
field_url = tk.Entry(window, width=70)
field_url.pack(padx=10, pady=10)

# LIMIT
label_limit = tk.Label(window, text="Limit liczby komentarzy (domyślnie 10):")
label_limit.pack(padx=10, pady=(10,0))
field_limit = tk.Entry(window, width=10)
field_limit.pack(padx=10, pady=10)

# PARAMETRY
vect_name_var = tk.StringVar(value='Bag of Words')  # Domyślna wartość
model_name_var = tk.StringVar(value='Logistic Regression')
lemmatization_var = tk.StringVar(value='precise')

# FRAME
frame_options = tk.Frame(window)
frame_options.pack(padx=10, pady=(15,0))

# FRAME VECTORIZER
frame_vect_name = tk.Frame(frame_options)
frame_vect_name.pack(side=tk.LEFT, padx=(0, 100))

label_vect_name = tk.Label(frame_vect_name, text="Wybierz metodę wektoryzacji:")
label_vect_name.pack()
tk.Radiobutton(frame_vect_name, text="Bag of Words", variable=vect_name_var, value="Bag of Words").pack()
tk.Radiobutton(frame_vect_name, text="TF-IDF", variable=vect_name_var, value="TF-IDF").pack()
tk.Label(frame_vect_name, text="").pack()

# FRAME MODEL
frame_model_name = tk.Frame(frame_options)
frame_model_name.pack(side=tk.LEFT, padx=(0, 100))

label_model_name = tk.Label(frame_model_name, text="Wybierz model:")
label_model_name.pack()
tk.Radiobutton(frame_model_name, text="Logistic Regression", variable=model_name_var, value="Logistic Regression").pack()
tk.Radiobutton(frame_model_name, text="SVM", variable=model_name_var, value="SVM").pack()
tk.Radiobutton(frame_model_name, text="Naive-Bayes", variable=model_name_var, value="Naive-Bayes").pack()

# FRAME LEMMATIZATION
frame_lemmatization = tk.Frame(frame_options)
frame_lemmatization.pack(side=tk.LEFT)

label_lemmatization = tk.Label(frame_lemmatization, text="Wybierz dokładność lematyzacji:")
label_lemmatization.pack()
tk.Radiobutton(frame_lemmatization, text="Precyzyjna", variable=lemmatization_var, value="precise").pack()
tk.Radiobutton(frame_lemmatization, text="Szybka", variable=lemmatization_var, value="quick").pack()
tk.Label(frame_lemmatization, text="").pack()


# SKANUJ KOMENTARZE
scan_button = tk.Button(window, text="Skanuj komentarze", command=scan_comment_section)
scan_button.pack(pady=10)

# PASEK PRZEWIJANIA
frame_table = tk.Frame(window)
frame_table.pack(expand=True, fill='both', padx=10, pady=10)

# TABELA - WYSWIETLA WYNIKI
table = ttk.Treeview(frame_table, columns=("Comment", "IsToxic"), show='headings')
table.column("Comment", width=int(window_width * 0.9), anchor='w')
table.column("IsToxic", width=int(window_width * 0.1), anchor='center')
table.heading("Comment", text="Komentarz")
table.heading("IsToxic", text="Toksyczność")
table.pack(side=tk.LEFT, fill='both', expand=True)

# PASEK PRZEWIJANIA
pasek_przewijania = tk.Scrollbar(frame_table, orient="vertical", command=table.yview)
pasek_przewijania.pack(side=tk.RIGHT, fill='y')
table.configure(yscrollcommand=pasek_przewijania.set)

# ZMIANA ROZMIARU OKNA
table.bind("<Configure>", update_columns)

# JAZDA
window.mainloop()
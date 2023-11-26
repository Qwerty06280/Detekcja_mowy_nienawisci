import tkinter as tk
#from tkinter import scrolledtext
from tkinter import ttk
import pandas as pd
from YT_API import yt_extract


# def analyze_comments(url: str):
#     """
#     Przeprowadza analizę komentarzy
#     :param url: link do filmu na YouTube
#     :return: dataframe, oprócz kolumny "Comment" zawiera kolumnę "IsToxic" z przewidzianywaną wartością
#     """
#     komentarz = ["test test :))))","chuju","dzielenie przez 0"]
#     isToxic = [0,1,1]
#     df = pd.DataFrame({"Comment": komentarz, "IsToxic": isToxic})
#     return df

def scan_comment_section():
    """

    :return:
    """
    url = field_url.get()
    try:
        limit = int(field_limit.get())
    except ValueError:
        limit = 10
        print(f"No limit / Wrong input -> limit has been set to {limit}")
    yt_comments, title = yt_extract(url = url, limit = limit)
    # Czyszczenie tabeli
    for i in table.get_children():
        table.delete(i)
    # Dodawanie nowych wierszy do tabeli
    for _, wiersz in yt_comments.iterrows():
        #toksycznosc = 'TOKSYCZNY' if wiersz['IsToxic'] else 'Bezpieczny'
        IsToxic = 'tbd'
        table.insert("", 'end', values=(wiersz['Comment'], IsToxic))

def update_columns(event):
    """
    Function for keeping the right proportion for displayed columns in tabble
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
label_url = tk.Label(window, text="URL filmu na YouTube:")
label_url.pack(padx=10, pady=(10,0))
# Pole do wprowadzenia URL
field_url = tk.Entry(window, width=70)
field_url.pack(padx=10, pady=10)

# LIMIT
label_limit = tk.Label(window, text="Limit liczby komentarzy (domyślnie 10):")
label_limit.pack(padx=10, pady=(10,0))
field_limit = tk.Entry(window, width=10)
field_limit.pack(padx=10, pady=10)

# Dodaj przycisk do skanowania komentarzy
scan_button = tk.Button(window, text="Skanuj komentarze", command=scan_comment_section)
scan_button.pack(pady=10)


# Ramka dla tabeli i paska przewijania
frame_table = tk.Frame(window)
frame_table.pack(expand=True, fill='both', padx=10, pady=10)

# Tabela do wyświetlania wyników
table = ttk.Treeview(frame_table, columns=("Comment", "IsToxic"), show='headings')
table.column("Comment", width=int(window_width * 0.9), anchor='w')
table.column("IsToxic", width=int(window_width * 0.1), anchor='center')
table.heading("Comment", text="Komentarz")
table.heading("IsToxic", text="Toksyczność")
table.pack(side=tk.LEFT, fill='both', expand=True)

# Pasek przewijania
pasek_przewijania = tk.Scrollbar(frame_table, orient="vertical", command=table.yview)
pasek_przewijania.pack(side=tk.RIGHT, fill='y')
table.configure(yscrollcommand=pasek_przewijania.set)

# Reakcja na zmianę rozmiaru okna
table.bind("<Configure>", update_columns)

# Jazda
window.mainloop()
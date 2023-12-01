import tkinter as tk
#from tkinter import scrolledtext
from tkinter import ttk
from YT_API import yt_extract
from yt_detection import youtube_detection

def scan_comment_section():
    """
    Call this function when the user clicks 'Skanuj komentarze'
    Takes input from user, downloads comments from youtube, analyzes them and displays the results
    """
    url = field_url.get()
    vect_name = vect_name_var.get()
    model_name = model_name_var.get()
    lemmatization = lemmatization_var.get()
    # LIMIT
    try:
        limit = int(field_limit.get())
    except ValueError:
        limit = 10
    # POPRAWNOŚĆ LINKU URL
    try:
        yt_comments, title = yt_extract(url=url, limit=limit)
        label_wrong_url.config(text="")
        label_title.config(text="Tytuł filmu: "+title)
    except:
        label_wrong_url.config(text="Wystąpił błąd podczas pobierania komentarzy z YouTube. Sprawdź poprawność podanego adresu URL.")
        return 0
    predicted = youtube_detection(yt_comments, vect_name= vect_name, model_name= model_name, lemmatization= lemmatization)
    yt_comments['IsToxic'] = predicted
    for i in table.get_children():
        table.delete(i)
    for col, row in yt_comments.iterrows():
        if row['IsToxic'] == 1:
            IsToxic = 'TOKSYCZNY'
        else:
            IsToxic = 'Bezpieczny'
        toxic_comment = table.insert("", 'end', values=(row['Comment'], IsToxic))
        if IsToxic == 'TOKSYCZNY':
            table.item(toxic_comment, tags=('bold','red_color'))

# set font and background for the interface here
font = "Georgia"
background = '#FFFBF5'

# Tkinter start
window = tk.Tk()
window.title("Detektor mowy nienawiści na polskojęzycznym YouTube")
window.configure(bg=background)

# Rozmiar okna
window_height = 800
window_width = 1200
window.geometry(f"{window_width}x{window_height}")

# URL
label_url = tk.Label(window, text="Link do filmu na portalu YouTube:", font=(font, 16, "bold"), background=background)
label_url.pack(padx=10, pady=(2,0))
field_url = tk.Entry(window, width=70)
field_url.pack(padx=10, pady=2)

# ERROR
label_wrong_url = tk.Label(window, text="", fg="red", font=((font, 8)), background=background)
label_wrong_url.pack(padx=2, pady=1)

# FRAME - limit and title
frame_limit_title = tk.Frame(window, background=background)
frame_limit_title.pack(padx=10, pady=(8,15))

# LIMIT
label_limit = tk.Label(frame_limit_title, text="Limit liczby komentarzy (domyślnie 10):", font=(font, 10, 'italic'),
                       background=background, borderwidth=3)
label_limit.pack(side=tk.LEFT, padx=(215, 5))
field_limit = tk.Entry(frame_limit_title, width=10)
field_limit.pack(side=tk.LEFT, padx=(0, 100))

# TITLE
label_title = tk.Label(frame_limit_title, text="Tytuł filmu: (brak tytułu)", width=400, wraplength=400, background=background,
                       font=(font, 10, 'italic'), anchor='w')
label_title.pack(side=tk.RIGHT, padx=(20, 50))

# PARAMETRY
vect_name_var = tk.StringVar(value='Bag of Words')  # Domyślna wartość
model_name_var = tk.StringVar(value='Logistic Regression')
lemmatization_var = tk.StringVar(value='precise')

# FRAME
frame_options = tk.Frame(window, background=background)
frame_options.pack(padx=10, pady=(8,0))

# FRAME VECTORIZER
frame_vect_name = tk.Frame(frame_options, background=background)
frame_vect_name.pack(side=tk.LEFT, padx=(0, 100))

label_vect_name = tk.Label(frame_vect_name, text="Wybierz metodę wektoryzacji:", font=(font, 12), background=background)
label_vect_name.pack()
tk.Radiobutton(frame_vect_name, text="Bag of Words", variable=vect_name_var, value="Bag of Words", font=(font, 10),background=background).pack()
tk.Radiobutton(frame_vect_name, text="TF-IDF", variable=vect_name_var, value="TF-IDF", font=(font, 10),background=background).pack()
tk.Label(frame_vect_name, text="", background=background).pack()

# FRAME MODEL
frame_model_name = tk.Frame(frame_options, background=background)
frame_model_name.pack(side=tk.LEFT, padx=(0, 100))

label_model_name = tk.Label(frame_model_name, text="Wybierz model:", font=(font, 12), background=background)
label_model_name.pack()
tk.Radiobutton(frame_model_name, text="Logistic Regression", variable=model_name_var, value="Logistic Regression", font=(font, 10), background=background).pack()
tk.Radiobutton(frame_model_name, text="SVM", variable=model_name_var, value="SVM", font=(font, 10), background=background).pack()
tk.Radiobutton(frame_model_name, text="Naive-Bayes", variable=model_name_var, value="Naive-Bayes", font=(font, 10), background=background).pack()

# FRAME LEMMATIZATION
frame_lemmatization = tk.Frame(frame_options, background=background)
frame_lemmatization.pack(side=tk.LEFT)

label_lemmatization = tk.Label(frame_lemmatization, text="Wybierz dokładność lematyzacji:",
                               font=(font, 12), background=background)
label_lemmatization.pack()
tk.Radiobutton(frame_lemmatization, text="Precyzyjna", variable=lemmatization_var, value="precise", font=(font, 10), background=background).pack()
tk.Radiobutton(frame_lemmatization, text="Szybka", variable=lemmatization_var, value="quick", font=(font, 10), background=background).pack()
tk.Label(frame_lemmatization, text="",background=background).pack()


# SKANUJ KOMENTARZE
def on_enter(e):
    scan_button.config(bg="lightgrey", fg="black")
def on_leave(e):
    scan_button.config(bg="#F8E8EE", fg="black")
scan_button = tk.Button(window, text="Skanuj komentarze", command=scan_comment_section,
                        font=(font, 12), bg="#F8E8EE", fg="black",relief=tk.RAISED, borderwidth=3)
scan_button.bind("<Enter>", on_enter)
scan_button.bind("<Leave>", on_leave)
scan_button.pack(pady=15)

# SCROLLING BAR
frame_table = tk.Frame(window)
frame_table.pack(expand=True, fill='both', padx=10, pady=10)

# TABLE - DISPLAYS RESULTS
def fixed_map(option):
    # Fix for setting text colour for Tkinter 8.6.9
    # From: https://core.tcl.tk/tk/info/509cafafae
    # Returns the style map for 'option' with any styles starting with
    # ('!disabled', '!selected', ...) filtered out.
    # style.map() returns an empty list for missing options, so this
    # should be future-safe.
    return [elm for elm in style.map('Treeview', query_opt=option) if
      elm[:2] != ('!disabled', '!selected')]
style = ttk.Style()
style.map('Treeview', foreground=fixed_map('foreground'),
  background=fixed_map('background'))
style.configure("Treeview", background="#fbfffa")

table = ttk.Treeview(frame_table, columns=("Comment", "IsToxic"), show='headings')
table.tag_configure('bold', font=('Helvetica', 10, 'bold')) # bold toxic comments
table.tag_configure('red_color', font=('Helvetica', 10, 'bold'), background='#fdecec') # bold toxic comments
table.column("Comment", width=int(window_width * 0.7), anchor='w')
table.column("IsToxic", width=int(window_width * 0.2), anchor='center')
table.heading("Comment", text="Komentarz")
table.heading("IsToxic", text="Toksyczność")
table.pack(side=tk.LEFT, fill='both', expand=True)
window.update_idletasks()

# SCROLLING BAR
pasek_przewijania = tk.Scrollbar(frame_table, orient="vertical", command=table.yview)
pasek_przewijania.pack(side=tk.RIGHT, fill='y')
table.configure(yscrollcommand=pasek_przewijania.set)

# ADJUSTING TO CHANGING WINDOW'S SIZE
def update_columns(event):
    """
    Function for keeping the right proportion for displayed columns in a table
    """
    table_width = table.winfo_width()
    is_toxic_width = 100
    table.column("Comment", width=int(table_width-is_toxic_width))
    table.column("IsToxic", width=int(is_toxic_width))
table.bind("<Configure>", update_columns)

# JAZDA
window.mainloop()
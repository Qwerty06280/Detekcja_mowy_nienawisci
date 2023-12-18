# Wykrywanie toksycznych/nienawistnych komentarzy - Praca inżynierska
### Autor: Bartosz Woziwoda
### Opis
Repozytorium zawiera wszystkie pliki wykorzystane podczas realizacji projektu dyplomowego pt. "Wykrywanie toksycznych/nienawistnych komentarzy".
Program można podzielić na 3 główne części:
- Łączenie z portalem YouTube za pomocą API
- Budowa klasyfikatorów, dzielących tekst na treść toksyczną i neutralną.
- Interfejs graficzny

Aplikacja pozwala użytkownikowi przeskanować sekcję komentarzy pod wybranym filmem na portalu YouTube w celu odszukania toksycznej treści. Interfejs prezentuje się następująco:
![image](https://github.com/Qwerty06280/Inzynierka/assets/62118154/ab4790cc-5021-403a-acc1-a3aee44e4622)

### Zawartość repozytorium
- Folder "Praca_inzynierska" - treść pracy dyplomowej
- Folder "app_screenshots" - zrzuty ekranu przedstawiające działanie aplikacji
- Folder "documentation" - dokumentacja kodu
- Folder "jupyter notebooks - development" - notatniki, w których rozwijane i testowane było oprogramowanie
- Folder "models_trained" - wyuczone modele NLP oraz obiekty wektoryzacji
- plik .travis.yml - konfiguracja usługi Travis CI
- plik CONCATENATED_DATA.xlsx - zbiór danych, na którym trenowano i testowano modele NLP
- pliki z rozszerzeniem .py - kod realizujący działanie aplikacji

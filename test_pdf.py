import fitz

def test_read_pdf(file_path):
    print(f"Пытаемся прочитать файл через PyMuPDF: {file_path}")
    try:
        doc = fitz.open(file_path)
        print(f"Всего страниц {len(doc)}")

        page = doc[0]
        text = page.get_text()

        print("--------Текст документа--------------")
        if text.strip():
            print(text)
        else:
            print("[Текст всё ещё не найден. Возможно, документ состоит из картинок (скан)]")
        print("=======================\n")
    except Exception as e:
        print(f"Произошла ошибка: {e}")



if __name__  == "__main__":
    pdf_path = r"C:\Projects\rag_project\data\tbank_tariff.pdf"
    test_read_pdf(pdf_path)
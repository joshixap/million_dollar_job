# encode_full_dataset.py

import pandas as pd
import numpy as np
import os
import re
import hashlib

def detect_gender(full_name):
    if pd.isna(full_name):
        return np.nan
    first = str(full_name).split()[0].strip().lower()
    return 0 if first.endswith(('а', 'я')) else 1

def extract_passport_year(passport_str):
    """Обрабатывает паспорта РФ, РБ, КЗ."""
    if pd.isna(passport_str):
        return np.nan
    s = str(passport_str).strip()
    
    # Россия: "6112 874629" → серия = "6112" → [2:4] = "12"
    if re.match(r'^\d{4}\s+\d{6}$', s):
        series = s[:4]
        if len(series) >= 4:
            return int(series[2:4])
    
    # Беларусь: "MC1234567" → цифры = "1234567" → последние 2 = "67"
    elif re.match(r'^[A-Z]{2}\d{7}$', s):
        digits = re.sub(r'\D', '', s)
        if len(digits) >= 2:
            return int(digits[-2:])
    
    # Казахстан: "123456789" (9 цифр)
    elif re.match(r'^\d{9}$', s):
        if len(s) >= 2:
            return int(s[:2])
    
    # Попытка извлечь любые 2 цифры в позициях 2-3 (fallback для РФ)
    digits_only = re.sub(r'\D', '', s)
    if len(digits_only) >= 4:
        return int(digits_only[2:4])
    
    return np.nan

def encode_snils(snils_str):
    if pd.isna(snils_str):
        return np.nan
    digits = re.sub(r'\D', '', str(snils_str))
    return int(digits) if digits and len(digits) <= 15 else np.nan

def normalize_item(item):
    return re.sub(r'\s+', ' ', str(item).strip().lower()) if pd.notna(item) else ""

def build_vocab(series):
    all_items = set()
    for val in series.dropna():
        if isinstance(val, str):
            parts = [normalize_item(x) for x in val.split(',') if x.strip()]
            all_items.update(parts)
        else:
            all_items.add(normalize_item(val))
    return {item: idx for idx, item in enumerate(sorted(all_items))}

def encode_multilabel(series, vocab):
    def _encode(val):
        if pd.isna(val):
            return 0
        if isinstance(val, str):
            items = [normalize_item(x) for x in val.split(',') if x.strip()]
        else:
            items = [normalize_item(val)]
        ids = sorted(vocab[item] for item in items if item in vocab)
        if not ids:
            return 0
        hash_val = hashlib.md5(str(ids).encode()).hexdigest()
        return int(hash_val, 16) % (2**32)
    return series.apply(_encode)

def encode_single_label(series, vocab):
    def _encode(val):
        if pd.isna(val):
            return -1
        item = normalize_item(val)
        return vocab.get(item, -1)
    return series.apply(_encode)

def extract_bin(card_str):
    if pd.isna(card_str):
        return np.nan
    clean = re.sub(r'\s+', '', str(card_str))
    digits = re.sub(r'\D', '', clean)
    return int(digits[:6]) if len(digits) >= 6 else np.nan

def extract_price(text):
    if pd.isna(text):
        return np.nan
    match = re.search(r'[\d\s]+', str(text))
    if match:
        num_str = match.group().replace(' ', '')
        try:
            return float(num_str)
        except:
            return np.nan
    return np.nan

def main():
    df = pd.read_csv('working_data/original_full.csv', sep=';', encoding='utf-8-sig')
    print(f"Загружено: {df.shape}")

    # 1. ФИО → пол
    df['Пол'] = df['ФИО'].apply(detect_gender)

    # 2. Паспорт → год (с поддержкой РБ, КЗ)
    df['Паспорт_год'] = df['Паспортные данные'].apply(extract_passport_year)

    # 3. СНИЛС → число
    df['СНИЛС_число'] = df['СНИЛС'].apply(encode_snils)

    # 4. Симптомы → хэш
    sym_vocab = build_vocab(df['Симптомы'])
    df['Симптомы_хэш'] = encode_multilabel(df['Симптомы'], sym_vocab)

    # 5. Врач → ID
    doc_vocab = build_vocab(df['Выбор врача'])
    df['Врач_ID'] = encode_single_label(df['Выбор врача'], doc_vocab)

    # 6. Дата визита → дни
    df['Дата посещения врача'] = pd.to_datetime(df['Дата посещения врача'], errors='coerce')
    min_visit = df['Дата посещения врача'].min()
    df['Дни_визита'] = (df['Дата посещения врача'] - min_visit).dt.days

    # 7. Анализы → хэш
    anal_vocab = build_vocab(df['Анализы'])
    df['Анализы_хэш'] = encode_multilabel(df['Анализы'], anal_vocab)

    # 8. Дата анализов → дни (без разности!)
    df['Дата получения анализов'] = pd.to_datetime(df['Дата получения анализов'], errors='coerce')
    min_anal = df['Дата получения анализов'].min()
    df['Дни_анализов'] = (df['Дата получения анализов'] - min_anal).dt.days

    # 9. Стоимость → число
    df['Стоимость_число'] = df['Стоимость анализов'].apply(extract_price)

    # 10. Карта → BIN
    df['Карта_BIN'] = df['Карта оплаты'].apply(extract_bin)

    # Финальные колонки (БЕЗ 'Разность_дат')
    final_cols = [
        'Пол', 'Паспорт_год', 'СНИЛС_число',
        'Симптомы_хэш', 'Врач_ID', 'Дни_визита',
        'Анализы_хэш', 'Дни_анализов',
        'Стоимость_число', 'Карта_BIN'
    ]
    final_df = df[final_cols].copy()

    os.makedirs('working_data', exist_ok=True)
    final_df.to_csv('working_data/encoded_full.csv', sep=';', encoding='utf-8-sig', index=False)
    print(f"\n Оцифровка завершена. Размер: {final_df.shape}")
    print("Сохранено: working_data/encoded_full.csv")

if __name__ == "__main__":
    main()
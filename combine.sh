#!/bin/bash

# Файл вывода
OUTPUT="combined_code.txt"

# Очищаем/создаём выходной файл
> "$OUTPUT"

# Файлы и директории, которые нужно исключить
EXCLUDE_DIRS=(".venv" "weights" "target")
EXCLUDE_FILES=("combined_code.txt")

# Функция проверки, нужно ли исключить путь
should_exclude() {
    local path="$1"
    
    # Проверяем исключаемые директории
    for dir in "${EXCLUDE_DIRS[@]}"; do
        if [[ "$path" == *"/$dir/"* ]] || [[ "$path" == *"/$dir" ]]; then
            return 0  # Исключаем
        fi
    done
    
    # Проверяем исключаемые файлы
    for file in "${EXCLUDE_FILES[@]}"; do
        if [[ "$(basename "$path")" == "$file" ]]; then
            return 0  # Исключаем
        fi
    done
    
    return 1  # Не исключаем
}

# Ищем все файлы в crates/hard-kernel-bench (рекурсивно)
find crates/smart-contracts-kernel-bench -type f | sort | while read -r filepath; do
    # Пропускаем исключаемые пути
    if should_exclude "$filepath"; then
        continue
    fi
    
    # Добавляем заголовок с именем файла
    echo "// $filepath" >> "$OUTPUT"
    echo "" >> "$OUTPUT"
    
    # Добавляем содержимое файла
    cat "$filepath" >> "$OUTPUT"
    
    # Добавляем разделитель между файлами
    echo "" >> "$OUTPUT"
    echo "// ============================================" >> "$OUTPUT"
    echo "" >> "$OUTPUT"
done

echo "Готово! Результат сохранён в: $OUTPUT"
echo "Всего обработано файлов: $(grep -c '^// ' "$OUTPUT")"
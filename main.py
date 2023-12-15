#!/usr/bin/env python
""" \
     Виконав студент 5 курсу Сербенюк Олександр
     Опис: 
     Даний код працює з файлом розширення JSONL із даними тіла подій Github
     Наразі обробляються дані типу PushEvent, витягує всі повідомлення комітів
     і перетворює розділені слова 3-грами.
"""

# Ліби
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    lower, explode,
    regexp_replace,
    collect_list,
    split, flatten,
    size, concat_ws,
    expr,
)
from pyspark.ml.feature import NGram
from time import strftime

# Визначаємо вхідні параметри
eventType = "PushEvent"
inputFilePath = "10K.github.jsonl"
ngramFactor = 3
ngramsColumnName = "ngrams"

# Конфігуруємо об'єкт для подальших калькулацій
ctx = SparkContext.getOrCreate()
spark = (SparkSession(ctx).builder.master("local[*]").appName("lab_1").getOrCreate())
ngram = NGram(n=ngramFactor, inputCol="words", outputCol=ngramsColumnName)

# Отримаємо повідомлення комітів та конвертуємо їх у колекцію AuthorName:CommitMessageWords
githubEventsDf = (spark.read.json(inputFilePath).filter(f"type = '{eventType}'")
        .select(explode("payload.commits").alias("commit"))
        .select(lower("commit.author.name").alias("author"), lower("commit.message").alias("message"))
        .withColumn("message", regexp_replace("message", "[^a-zA-Z0-9\\s]", ""))
        .withColumn("message", (split("message", "\\s+")))
        .withColumn("message", expr("filter(message, element -> element != '')"))
        .groupBy("author")
        .agg(flatten(collect_list("message")).alias("words")))

# Обробляємо слова в n-grams
result = (ngram.transform(githubEventsDf)
        .select("author", ngramsColumnName)
        .filter(size(ngramsColumnName) > 0)
        .withColumn(ngramsColumnName, concat_ws(", ", ngramsColumnName)))

# Зберігаємо результат в csv файл
resultFileName = "ngram-{timestamp}".format(timestamp=strftime("%Y%m%d-%H%M%S"))
result.write.option("header", True).option("delimiter", "|").csv(resultFileName)

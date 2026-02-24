Ты работаешь в корне репозитория IDEA_PIPELINE_DIST. Реализуй Stage B “Multi-Source Literature Scout” по SPEC (см. SPEC_STAGE_B_RU.md).

Важные условия:
- Windows (PowerShell/.bat), кодировка UTF-8, русские сообщения.
- Источники: OpenAlex + NCBI (ключи уже в config/secrets.env). Дополнительно реализуй коннекторы (плагины) Crossref, Unpaywall, EuropePMC, arXiv, bioRxiv/medRxiv как опциональные: если нет доступа/ошибка/нет ключа — НЕ падать, пометить в логах и продолжить.
- Итог в консоли: краткая сводка OK/DEGRADED/FAILED + числа по источникам + путь к corpus.csv, prisma_lite.md. И сохранить ту же сводку в out/stageB_summary.txt.
- Универсальность: не использовать доменные словари “под рыбу”; генерация запросов должна работать для любых тем.
- Дедуп: DOI>PMID>OpenAlexID>arXivID>title+year.
- Артефакты: corpus.csv, search_log.json, prisma_lite.md, field_map.md, module_B.log, checkpoint.

Сделай так:
1) Сначала изучи существующую структуру проекта, как устроены Stage A и текущая Stage B.
2) Внеси минимально рискованные изменения.
3) Добавь smoke-тест: прогон RUN_B на 2–3 тест-идеях (папки ideas/IDEA-... уже есть) и убедись, что:
   - при отключении одного источника всё равно OK/DEGRADED и есть corpus.csv
   - повторный запуск использует checkpoint и не делает лишних запросов
4) В конце дай список изменённых файлов и кратко: как проверить вручную.
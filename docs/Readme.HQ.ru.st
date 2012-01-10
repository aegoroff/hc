$langName$ Readme 

Что такое $langName$? Если коротко, то это интерпретатор (или компилятор, если угодно) декларативного языка запросов на вычисление криптографических хэшей для строк и файлов. $langName$ умеет:

- вычислять хэш строки
- вычислять хэш файла, в том числе и его часть (задаваемой смещением и длиной)
- восстанавливать строку по хэшу методом грубой силы (перебор по конечному словарю)
- вычислять хэш для файлов в заданном каталоге, с поддержкой фильтрации файлов по размеру, имени, пути
- проверять файл с использованием известного хэша (валидация файла)
- искать файлы с использованием известного хэша всего файла или его части

Кроме того есть:

- Оценка времени восстановления строки с использованием грубой силы
- Вывод хэша в разном регистре
- Поддержка запросов из файлов и из командной строки
- Поддержка комментариев в файлах с запросами

Поддерживается следующие типы хэшей:

- MD5
- MD4
- SHA1
- SHA256
- SHA384
- SHA512
- Whirlpool
- CRC32

Использование

$appName$ [OPTION] ...

Доступные опции:

-f [ --file ] <путь>      Полный путь к файлу c запросами на языке $langName$
-q [ --query ] <текст>	  Текст запроса на языке $langName$
-v [ --validate ]         Проверить синтаксис запроса. Не запускать никаких действий
-t [ --time ]             Показывать время вычисления хэша (по умолчанию отключено)
-l [ --lower ]            Выводить хэш в нижнем регистре
-? [ --help ]             Показать помощь


Примеры

Вычисление SHA1 хэша для строки 123

$appName$.exe -q "for string '123' do sha1;"


Вычисление MD5 хэша для файла

$appName$.exe -q "for file f from 'file.txt' do md5;"


Вычисление SHA384 хэша для части файла (первый килобайт)

$appName$.exe -q "for file f from 'file.txt' let f.limit = 1024 do sha384;"


Вычисление SHA256 хэша части файла (один килобайт с пропуском первых 512 байт)

$appName$.exe -q "for file f from 'file.txt' let f.limit = 1024, f.offset = 512 do sha256;"


Валидация файла по его MD4 хэшу

$appName$.exe -q "for file f from 'file.txt' let f.md4 = 'C58CDA49F00748A3BC0FCFA511D516CB' do validate;"


Вычисление SHA512 хэша всех файлов каталога c:\dir

$appName$.exe -q "for file f from dir 'c:\dir' do sha512;"


Вычисление Whirlpool хэша всех файлов каталога c:\dir а также всех его подкаталогов

$appName$.exe -q "for file f from dir 'c:\dir' do whirlpool withsubs;"


Вычисление CRC32 суммы всех exe файлов каталога c:\dir

$appName$.exe -q "for file f from dir 'c:\dir' where f.name ~ '.*exe$' do crc32;"


Вычисление MD5 хэша всех файлов каталога c:\dir кроме файлов с расширением tmp

$appName$.exe -q "for file f from dir 'c:\dir' where f.name !~ '.*tmp$' do md5;"


Вычисление MD5 хэша всех exe и dll файлов каталога c:\dir

$appName$.exe -q "for file f from dir 'c:\dir' where f.name ~ '.*exe$' or f.name ~ '.*dll$' do md5;"


Вычисление MD5 хэша всех exe файлов каталога c:\dir исключая те из них, которые начинаются с bad

$appName$.exe -q "for file f from dir 'c:\dir' where f.name !~ '^bad.*' do md5;"


Поиск файла на диске C:\ по известному MD4 хэшу

$appName$.exe -q "for file f from dir 'c:\' where f.md4 == 'C58CDA49F00748A3BC0FCFA511D516CB' do find withsubs;"


Восстановление строки по её MD4 хэшу используя словарь по умолчанию

$appName$.exe -q "for string s from hash '3689CA24BF71B39B6612549D87DCEA68' do crack md4;"


Восстановление строки по её MD4 хэшу используя свой словарь

$appName$.exe -q "for string s from hash '3689CA24BF71B39B6612549D87DCEA68' let s.dict = '0123456789' do crack md4;"


Восстановление строки по её MD4 хэшу используя свой словарь сокращенная форма (одни цифры)

$appName$.exe -q "for string s from hash '3689CA24BF71B39B6612549D87DCEA68' let s.dict = '0-9' do crack md4;"


Восстановление строки по её MD4 хэшу используя свой словарь сокращенная форма (одни буквы в нижнем регистре)

$appName$.exe -q "for string s from hash '3689CA24BF71B39B6612549D87DCEA68' let s.dict = 'a-z' do crack md4;"


Восстановление строки по её MD4 хэшу используя свой словарь сокращенная форма (цифры, буквы в нижнем и верхнем регистрах)

$appName$.exe -q "for string s from hash '3689CA24BF71B39B6612549D87DCEA68' let s.dict = '0-9a-zA-Z' do crack md4;"


Восстановление строки по её MD4 хэшу используя свой словарь и определенные длины строки

$appName$.exe -q "for string s from hash '3689CA24BF71B39B6612549D87DCEA68' let s.dict = '0123456789', s.min = 2, s.max = 6 do crack md4;"
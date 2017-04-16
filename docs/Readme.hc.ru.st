$langName$ Readme 

$langName$ это консольное приложение, вычисляющиее около 50 различных криптографических хэшей строк и файлов. $langName$ умеет:

- вычислять хэш строки
- вычислять хэш файла, в том числе и его часть (задаваемой смещением и длиной)
- восстанавливать строку по хэшу методом грубой силы (перебор по конечному словарю)
- вычислять хэш для файлов в заданном каталоге, с поддержкой фильтрации файлов по размеру, имени, пути
- проверять файл с использованием известного хэша (валидация файла)
- искать файлы с использованием известного хэша всего файла или его части

Кроме того есть:

- Оценка времени восстановления строки с использованием грубой силы
- Многопоточное восстановление строки по её хэшу
- Вывод хэша в разном регистре
- Вывод результатов вычислений в формате SFV (simple file verification)

Поддерживается следующие типы хэшей:

- MD2
- MD4
- MD5
- SHA1
- SHA224
- SHA256
- SHA384
- SHA512
- SHA3 224 (FIPS 202)
- SHA3 256 (FIPS 202)
- SHA3 384 (FIPS 202)
- SHA3 512 (FIPS 202)
- SHA3 224 (Keccak)
- SHA3 256 (Keccak)
- SHA3 384 (Keccak)
- SHA3 512 (Keccak)
- Whirlpool
- Ripemd 128
- Ripemd 160
- Ripemd 256
- Ripemd 320
- Tiger-192
- Tiger2-192
- CRC32
- ГОСТ 34.11-94
- Snerfu 128
- Snerfu 256
- TTH (Tiger Tree Hash)
- HAVAL 128, 3
- HAVAL 128, 4
- HAVAL 128, 5
- HAVAL 160, 3
- HAVAL 160, 4
- HAVAL 160, 5
- HAVAL 192, 3
- HAVAL 192, 4
- HAVAL 192, 5
- HAVAL 224, 3
- HAVAL 224, 4
- HAVAL 224, 5
- HAVAL 256, 3
- HAVAL 256, 4
- HAVAL 256, 5
- EDON-R 256
- EDON-R 512
- NTLM
- BLAKE 2b
- BLAKE 2s


Использование

Режим вычисления хэша строки

$appName$ [-lh] <algorithm> string -s <string>

  <algorithm>               хэш алгоритм. См. возможные значения ниже
  -s, --string=<string>     строка для которой вычисляется хэш
  -l, --lower               выход хэша с использованием нижнего регистра (по умолчанию  в верхнем)
  -h, --help                показать помощь и выйти

  
Режим восстановления хэша строки

$appName$ [-bplh] <algorithm> hash [-m <string>] [-a <string>] [-n <int>] [-x <int>] [--noprobe] [-T <int>]

  <algorithm>               хэш алгоритм. См. возможные значения ниже
  -m, --hash=<string>       хэш по которому будет восстанавливаться строка
  -b, --base64hash          хэш задан в форме base64
  -a, --dict=<string>       словарь строки. Все цифры, латинские буквы в верхнем и нижнем регистре по умолчанию
  -n, --min=<int>           минимальная длина строки для восстановления. По умолчанию 1
  -x, --max=<int>           set maximum length of the string to restore  using 
                            option crack (c). 10 by default
  -p, --performance         тест производительности путем восстановления  хэша 
                            строки 12345
      --noprobe             не оценивать время восстановления строки
  -T, --threads=<int>       количество потоков, чтобы взломать хэш. Половина количества процессоров системы по умолчанию. Значение
                            должно быть между 1 и числом равным количеству процессоров.
  -l, --lower               выход хэша с использованием нижнего регистра (по умолчанию  в верхнем)
  -h, --help                показать помощь и выйти

  
Режим вычисления хэша файлов

$appName$ [-ctlh] <algorithm> file -f <file> [-m <string>] [-z <number>] [-q <number>] [-o <file>] [--sfv]

  <algorithm>               хэш алгоритм. См. возможные значения ниже
  -f, --file=<file>         полный путь к файлу для вычисления хэш-суммы
  -m, --hash=<string>       хэш для проверки файла
  -z, --limit=<number>      количество байт файла для вычисления хэша. По умолчанию будет использоваться весь файл
  -q, --offset=<number>     стартовая позиция в файле для вычисления хэша. По умолчанию начало файла
  -c, --checksumfile        вывод формате файла контрольной суммы
  -o, --save=<file>         сохранение хэшей файлов в указанный файл, вместо консоли.
  -t, --time                показывать время вычислений (по умолчанию нет)
      --sfv                 выводить в виде SFV (Simple File Verification). Только для CRC32. По умолчанию нет.
  -l, --lower               выход хэша с использованием нижнего регистра (по умолчанию  в верхнем)
  -h, --help                показать помощь и выйти

  
Режим вычисления хэша файлов в каталогах

$appName$ [-rctlh] <algorithm> dir -d <string> [-m <string>] [-e <string>] [-i <string>] [-z <number>] [-q <number>] [-H <string>] [-o <file>] [--sfv] [--noerroronfind]

  <algorithm>               хэш алгоритм. См. возможные значения ниже
  -d, --dir=<string>        полный путь к каталогу для вычисления хэша всех его
                            файлов
  -m, --hash=<string>       хэш для проверки файлов в каталоге
  -e, --exclude=<string>    exclude files that match the pattern specified. It'
                            s possible to use several patterns separated by ;
  -i, --include=<string>    include only files that match the pattern specified
                            . It's possible to use several patterns separated b
                            y ;
  -z, --limit=<number>      количество байт файла для вычисления хэша. По умолчанию будет использоваться весь файл
  -q, --offset=<number>     стартовая позиция в файле для вычисления хэша. По умолчанию начало файла
  -H, --search=<string>     хэш для поиска файла который ему соответствует
  -r, --recursively         включая вложенные каталоги
  -c, --checksumfile        вывод формате файла контрольной суммы
  -o, --save=<file>         сохранение хэшей файлов в указанный файл, вместо ко
                            нсоли.
  -t, --time                показывать время вычислений (по умолчанию нет)
      --sfv                 выводить в виде SFV (Simple File Verification). Тол
                            ько для CRC32. По уполчанию нет.
  -l, --lower               выход хэша с использованием нижнего регистра (по умолчанию  в верхнем)
      --noerroronfind       не выводить ошибки во время поиска файлов. По умолчанию выводить
  -h, --help                показать помощь и выйти

  Все возможные хэш алгоритмы:
    blake2b
    blake2s
    crc32
    edonr256
    edonr512
    gost
    haval-128-3
    haval-128-4
    haval-128-5
    haval-160-3
    haval-160-4
    haval-160-5
    haval-192-3
    haval-192-4
    haval-192-5
    haval-224-3
    haval-224-4
    haval-224-5
    haval-256-3
    haval-256-4
    haval-256-5
    md2
    md4
    md5
    ntlm
    ripemd128
    ripemd160
    ripemd256
    ripemd320
    sha-3-224
    sha-3-256
    sha-3-384
    sha-3-512
    sha-3k-224
    sha-3k-256
    sha-3k-384
    sha-3k-512
    sha1
    sha224
    sha256
    sha384
    sha512
    snefru128
    snefru256
    tiger
    tiger2
    tth
    whirlpool

Примеры

Вычисление SHA1 хэша строки 123

$appName$.exe sha1 string -s 123


Вычисление SHA1 хэша файла

$appName$.exe sha1 file -f file.txt


Вычисление SHA1 хэша части файла (первый килобайт)

$appName$.exe sha1 file -f file.txt -z 1024


Вычисление SHA1 хэша части файла (один килобайт с пропуском первых 512 байт)

$appName$.exe sha1 file -f file.txt -z 1024 -q 512


Валидация файла по его SHA1 хэшу

$appName$.exe sha1 file -f file.txt -m 8BD7954C40C1E59A900F71EA3A266732609915B1


Вычисление SHA1 хэша всех файлов каталога c:\dir

$appName$.exe sha1 dir -d c:\dir


Вычисление SHA1 хэша всех файлов каталога c:\dir а также всех его подкаталогов

$appName$.exe sha1 dir -r -d c:\dir


Вычисление SHA1 хэша всех файлов каталога c:\dir и вывести результаты в формате файла проверки по контрольным суммам

$appName$.exe sha1 dir -d c:\dir --checksumfile


Вычисление SHA1 хэша всех файлов каталога c:\dir и сохранить в файл dir.sha1 результаты в формате файла проверки по контрольным суммам

$appName$.exe sha1 dir -d c:\dir --checksumfile -o dir.sha1


Вычисление CRC32 хэша всех файлов каталога c:\dir и сохранение результатов в формате CFV в файл dir.sfv

$appName$.exe crc32 dir -d c:\dir --sfv -o dir.sfv


Вычисление SHA1 хэша всех exe файлов каталога c:\dir

$appName$.exe sha1 dir -d c:\dir -i *.exe


Вычисление SHA1 хэша всех файлов каталога c:\dir кроме файлов с расширением tmp

$appName$.exe sha1 dir -d c:\dir -e *.tmp


Вычисление SHA1 хэша всех exe и dll файлов каталога c:\dir

$appName$.exe sha1 dir -d c:\dir -i *.exe;*.dll


Вычисление SHA1 хэша всех exe файлов каталога c:\dir исключая те из них, которые начинаются с bad

$appName$.exe sha1 dir -d c:\dir -i *.exe -e bad*


Поиск файла на диске C:\ по известному SHA1 хэшу

$appName$.exe sha1 dir -d c:\ -r -H 8BD7954C40C1E59A900F71EA3A266732609915B1


Восстановление строки по её SHA1 хэшу используя словарь по умолчанию

$appName$.exe sha1 hash -с -m 40BD001563085FC35165329EA1FF5C5ECBDBBEEF


Восстановление строки по её SHA1 хэшу используя свой словарь

$appName$.exe sha1 hash -с -m 40BD001563085FC35165329EA1FF5C5ECBDBBEEF -a 0123456789


Восстановление строки по её SHA1 хэшу используя свой словарь в краткой форме (только цифры)

$appName$.exe sha1 hash -с -m 40BD001563085FC35165329EA1FF5C5ECBDBBEEF -a 0-9


Восстановление строки по её SHA1 хэшу используя свой словарь в краткой форме (только латинские буквы в нижнем регистре)

$appName$.exe sha1 hash -с -m 40BD001563085FC35165329EA1FF5C5ECBDBBEEF -a a-z


Восстановление строки по её SHA1 хэшу используя свой словарь в краткой форме (цифры, латинские буквы в нижнем и верхнем регистрах)

$appName$.exe sha1 hash -с -m 40BD001563085FC35165329EA1FF5C5ECBDBBEEF -a 0-9a-zA-Z


Восстановление строки по её SHA1 хэшу используя свой словарь и определенные длины строки

$appName$.exe sha1 hash -с -m 40BD001563085FC35165329EA1FF5C5ECBDBBEEF -a 0123456789 -n 2 -x 6


Тест производительности с использованием словаря по умолчанию

$appName$.exe sha1 hash -p


Тест производительности с использованием своего словаря в краткой форме (только цифры)

$appName$.exe sha1 hash -p -a 0-9

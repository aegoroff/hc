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
- Многопоточное восстановление строки по её хэшу
- Вывод хэша в разном регистре
- Вывод результатов вычислений в формате SFV (simple file verification)
- Поддержка запросов из файлов и из командной строки
- Поддержка комментариев в файлах с запросами
- Поддержка переменных

Поддерживается следующие типы хэшей:

- MD2
- MD4
- MD5
- SHA1
- SHA224
- SHA256
- SHA384
- SHA512
- Whirlpool
- Ripemd 128
- Ripemd 160
- Ripemd 256
- Ripemd 320
- Tiger-192
- Tiger2-192
- CRC32
- ГОСТ
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


Использование

$appName$ [-rcpStlh] [<string>] [-f <file>] [-d <string>] [-e <string>] [-i <string>] [-s <string>] [-m <stri
ng>] [-a <string>] [-n <int>] [-x <int>] [-z <number>] [-q <number>] [-H <string>] [-o <file>] [-C <
string>] [-F <file>]... [-P <file>] [--sfv]

  <string>                  хэш алгоритм
  -f, --file=<file>         полный путь к файлу для которого вычислить хэш
  -d, --dir=<string>        полный путь к каталогу для всех файлов которого будет вычислен
                            заданный хэш
  -e, --exclude=<string>    маска (шаблон) файлов которые нужно исключать из обработки. 
                            Можно указать несколько масок разделяя их точкой с запятой ";".
                            Используется вместе с опцией -d (--dir)
  -i, --include=<string>    маска (шаблон) файлов которые нужно обрабатывать (остальные обрабатываться не будут). 
                            Можно указать несколько масок разделяя их точкой с запятой ";". 
                            Используется вместе с опцией -d (--dir)
  -s, --string=<string>     строка, для которой нужно подсчитать заданный хэш
  -m, --hash=<string>       хэш по которому нужно либо проверить файл (задается с опцией -f) или восстановить исходную строку (необходима опция -c)
  -a, --dict=<string>       словарь, по которому будет производиться восстановление строки по её хэшу. По умолчанию все цифры,
                            латинские буквы в верхнем и нижнем регистрах.
  -n, --min=<int>           минимальная длина строки для восстановления. По умолчанию 1
  -x, --max=<int>           максимальная длина строки для восстановления. По умолчанию 10
  -z, --limit=<number>      размер части файла в байтах для которой вычислять хэш. По умолчанию для всего файла
  -q, --offset=<number>     стартовая позиция в файле начиная с которой вычислять хэш. По умолчанию ноль
  -H, --search=<string>     хэш для поиска файла соответствующего ему
  -o, --save=<file>         сохранить хэши файлов в указанный файл.
  -r, --recursively         сканировать вложенные каталоги
  -c, --crack               восстановить указанный хэш (найти захэшированную строку) заданный опцией --hash (-m)
  -p, --performance         тест производительности путем восстановления хэша для строки 123
  -C, --command=<string>    текст запроса на встроенном языке переданный через командную строку
  -F, --query=<file>        один или несколько файлов с текстами запросов на встроенном языке
  -P, --param=<file>        путь к проверяемому файлу который будет проверен с помощью одного или нескольких запросов
  -S, --syntaxonly          только проверить синтаксис запроса на встроенном языке. Не запускать никаких действий
  -t, --time                показывать время вычисления хэша (по умолчанию отключено)
  -l, --lower               выводить хэш в нижнем регистре
      --sfv                 выводить хэш в формате SFV (Simple File Verification)
                            актуально только для вычисления хэша файлов в каталоге
      --noprobe             Отключить оценку времени восстановления строки по хэшу
  -T, --threads=<int>       количество потоков для восстановления строки по хэшу. По умолчанию равно половине количества процессоров(ядер) в системе. 
                            Может принимать значения от 1 до количества процессоров(ядер) в системе.
  -h, --help                вывести помощь и выйти


Формальная грамматика языка

prog
	: statement+ | EOF
	;
     
statement
    :   expr NEWLINE
    | NEWLINE
    ;

expr:
	FOR (expr_string | expr_hash | expr_dir | expr_file | expr_file_analyze) | expr_vardef
    ;

expr_vardef:
	LET ID ASSIGN_OP STRING
	;

expr_string
	: STR source DO hash_clause
	;


expr_hash:
	STR id FROM HASH source let_clause? DO brute_force_clause
	;

expr_dir
	: FILE id FROM DIR source ( file_hash_clause | file_find_clause ) (WITHSUBS)?
	;

file_hash_clause
    : let_clause? where_clause? DO hash_clause
    ;

file_find_clause
    : let_clause? where_clause DO FIND
    ;

expr_file
	: FILE id FROM source ( let_clause? DO hash_clause | let_clause DO VALIDATE )
	;

expr_file_analyze
	: FILE id FROM PARAMETER where_clause DO VALIDATE
	;

source : ID | STRING;

id : ID;

attr_clause : ID DOT attr ;

attr : str_attr | int_attr ;

hash_clause : ALG;
    
brute_force_clause
	: CRACK hash_clause
	;

let_clause
	: LET assign (COMMA assign)*
	;

where_clause
	: WHERE boolean_expression
    ;

boolean_expression
	: conditional_or_expression
	;

conditional_or_expression
	: conditional_and_expression (OR conditional_and_expression)*
	;

conditional_and_expression
	: not_expression (AND not_expression)* 
	;

not_expression
	: exclusive_or_expression
	| NOT_OP exclusive_or_expression
	;

exclusive_or_expression
	:	relational_expr
	|	OPEN_BRACE boolean_expression CLOSE_BRACE
	;

relational_expr
	: ID DOT ( relational_expr_str | relational_expr_int )
	;

relational_expr_str
	: str_attr 
	( EQUAL STRING
	| EQUAL ID
	| NOTEQUAL STRING
	| NOTEQUAL ID
	| MATCH STRING
	| MATCH ID
	| NOTMATCH STRING
	| NOTMATCH ID
	)
	;

relational_expr_int
	:	l=int_attr 
	(
	EQUAL INT
	| NOTEQUAL INT
	| GE INT
	| LE INT
	| LEASSIGN INT
	| GEASSIGN INT
	)
	;

assign 
	: ID DOT ( str_attr ASSIGN_OP STRING | str_attr ASSIGN_OP ID | int_attr ASSIGN_OP INT )
	;
 
str_attr
	: NAME_ATTR
	| PATH_ATTR
	| DICT_ATTR
	| ALG
	; 

int_attr
@init
	: SIZE_ATTR
	| LIMIT_ATTR
	| OFFSET_ATTR
	| MIN_ATTR
	| MAX_ATTR
	; 

ALG 
    : 'md2' 
    | 'md4' 
    | 'md5' 
    | 'sha1' 
    | 'sha224' 
    | 'sha256' 
    | 'sha384' 
    | 'sha512' 
    | 'crc32' 
    | 'whirlpool' 
    | 'tiger' 
    | 'tiger2' 
    | 'ripemd128' 
    | 'ripemd160' 
    | 'ripemd256' 
    | 'ripemd320' 
    | 'gost' 
    | 'snefru128' 
    | 'snefru256' 
    | 'tth' 
    | 'haval-128-3' 
    | 'haval-128-4' 
    | 'haval-128-5' 
    | 'haval-160-3' 
    | 'haval-160-4' 
    | 'haval-160-5' 
    | 'haval-192-3' 
    | 'haval-192-4' 
    | 'haval-192-5' 
    | 'haval-224-3' 
    | 'haval-224-4' 
    | 'haval-224-5' 
    | 'haval-256-3' 
    | 'haval-256-4' 
    | 'haval-256-5' 
    | 'edonr256' 
    | 'edonr512' 
    ;

NAME_ATTR :	'name';

PATH_ATTR :	'path' ;

DICT_ATTR :	'dict' ;

SIZE_ATTR :	'size' ;

LIMIT_ATTR :	'limit' ;

OFFSET_ATTR : 'offset' ;

MIN_ATTR : 'min' ;

MAX_ATTR : 'max' ;

CRACK :	'crack' ;

WHERE :	'where' ;

OR: 'or' ;

AND: 'and' ;

NOT_OP: 'not' ;

FOR: 'for' ;

FROM: 'from' ;

PARAMETER: 'parameter' ;

DO: 'do' ;

FIND: 'find' ;

WITHSUBS : 'withsubs' ;
VALIDATE : 'validate' ;

LET	: 'let' ;

DIR	:	'dir' ;
FILE	:	'file' ;
HASH	:	'hash' ;
STR	:	'string' ;

fragment
STRING1 : '\'' ( options {greedy=false;} : ~('\u0027' | '\u000A' | '\u000D'))* '\'' ;

fragment
STRING2 : '"'  ( options {greedy=false;} : ~('\u0022' | '\u000A' | '\u000D'))* '"' ;

STRING : STRING1 | STRING2 ;

ID : ID_START ID_PART* ;

fragment
ID_START : '_' | 'A'..'Z' | 'a'..'z' ;

fragment
ID_PART : ID_START | '0'..'9' ;

INT :   '0'..'9'+ ;
ASSIGN_OP : ASSIGN;

NEWLINE: ';';
WS  :   (' '|'\t'| EOL )+ ;
DOT	: '.' ;
COMMA: ',' ;	
OPEN_BRACE : '(';
CLOSE_BRACE : ')';

COMMENT : ('#' | '/' '/') ~(EOL)* CR? (LF | EOF) ;

fragment
EOL : LF | CR ;

fragment
LF :	'\n' ;

fragment
CR :	'\r' ;
 
PLUS:	'+' ;

EQUAL:	ASSIGN ASSIGN ;
NOTEQUAL:	NOT ASSIGN ;

fragment
ASSIGN:	'=' ;
fragment
NOT:	'!' ;

GE:	'>' ;
LE:	'<' ;
MATCH:	'~' ;
NOTMATCH : NOT MATCH ;
LEASSIGN :LE ASSIGN;
GEASSIGN :GE ASSIGN;


Примеры

Вычисление SHA1 хэша строки 123

$appName$.exe sha1 -s 123


Вычисление SHA1 хэша файла

$appName$.exe sha1 -f file.txt


Вычисление SHA1 хэша части файла (первый килобайт)

$appName$.exe sha1 -f file.txt -z 1024


Вычисление SHA1 хэша части файла (один килобайт с пропуском первых 512 байт)

$appName$.exe sha1 -f file.txt -z 1024 -q 512


Валидация файла по его SHA1 хэшу

$appName$.exe sha1 -f file.txt -m 8BD7954C40C1E59A900F71EA3A266732609915B1


Вычисление SHA1 хэша всех файлов каталога c:\dir

$appName$.exe sha1 -d c:\dir


Вычисление SHA1 хэша всех файлов каталога c:\dir а также всех его подкаталогов

$appName$.exe sha1 -r -d c:\dir


Вычисление SHA1 хэша всех exe файлов каталога c:\dir

$appName$.exe sha1 -d c:\dir -i *.exe


Вычисление SHA1 хэша всех файлов каталога c:\dir кроме файлов с расширением tmp

$appName$.exe sha1 -d c:\dir -e *.tmp


Вычисление SHA1 хэша всех exe и dll файлов каталога c:\dir

$appName$.exe sha1 -d c:\dir -i *.exe;*.dll


Вычисление SHA1 хэша всех exe файлов каталога c:\dir исключая те из них, которые начинаются с bad

$appName$.exe sha1 -d c:\dir -i *.exe -e bad*


Поиск файла на диске C:\ по известному SHA1 хэшу

$appName$.exe sha1 -d c:\ -r -h 8BD7954C40C1E59A900F71EA3A266732609915B1


Восстановление строки по её SHA1 хэшу используя словарь по умолчанию

$appName$.exe sha1 -с -m 40BD001563085FC35165329EA1FF5C5ECBDBBEEF


Восстановление строки по её SHA1 хэшу используя свой словарь

$appName$.exe sha1 -с -m 40BD001563085FC35165329EA1FF5C5ECBDBBEEF -a 0123456789


Восстановление строки по её SHA1 хэшу используя свой словарь в краткой форме (только цифры)

$appName$.exe sha1 -с -m 40BD001563085FC35165329EA1FF5C5ECBDBBEEF -a 0-9


Восстановление строки по её SHA1 хэшу используя свой словарь в краткой форме (только латинские буквы в нижнем регистре)

$appName$.exe sha1 -с -m 40BD001563085FC35165329EA1FF5C5ECBDBBEEF -a a-z


Восстановление строки по её SHA1 хэшу используя свой словарь в краткой форме (цифры, латинские буквы в нижнем и верхнем регистрах)

$appName$.exe sha1 -с -m 40BD001563085FC35165329EA1FF5C5ECBDBBEEF -a 0-9a-zA-Z


Восстановление строки по её SHA1 хэшу используя свой словарь и определенные длины строки

$appName$.exe sha1 -с -m 40BD001563085FC35165329EA1FF5C5ECBDBBEEF -a 0123456789 -n 2 -x 6


Тест производительности с использованием словаря по умолчанию

$appName$.exe sha1 -p


Тест производительности с использованием своего словаря в краткой форме (только цифры)

$appName$.exe sha1 -p -a 0-9


Вычисление SHA1 хэша для строки 123

$appName$.exe -c "for string '123' do sha1;"


Вычисление MD5 хэша для файла

$appName$.exe -c "for file f from 'file.txt' do md5;"


Вычисление SHA384 хэша для части файла (первый килобайт)

$appName$.exe -c "for file f from 'file.txt' let f.limit = 1024 do sha384;"


Вычисление SHA256 хэша части файла (один килобайт с пропуском первых 512 байт)

$appName$.exe -c "for file f from 'file.txt' let f.limit = 1024, f.offset = 512 do sha256;"


Валидация файла по его MD4 хэшу

$appName$.exe -c "for file f from 'file.txt' let f.md4 = 'C58CDA49F00748A3BC0FCFA511D516CB' do validate;"


Вычисление SHA512 хэша всех файлов каталога c:\dir

$appName$.exe -c "for file f from dir 'c:\dir' do sha512;"


Вычисление Whirlpool хэша всех файлов каталога c:\dir а также всех его подкаталогов

$appName$.exe -c "for file f from dir 'c:\dir' do whirlpool withsubs;"


Вычисление CRC32 суммы всех exe файлов каталога c:\dir

$appName$.exe -c "for file f from dir 'c:\dir' where f.name ~ '.*exe\$' do crc32;"


Вычисление MD5 хэша всех файлов каталога c:\dir кроме файлов с расширением tmp

$appName$.exe -c "for file f from dir 'c:\dir' where f.name !~ '.*tmp\$' do md5;"


Вычисление MD5 хэша всех exe и dll файлов каталога c:\dir

$appName$.exe -c "for file f from dir 'c:\dir' where f.name ~ '.*exe\$' or f.name ~ '.*dll\$' do md5;"


Вычисление MD5 хэша всех exe файлов каталога c:\dir исключая те из них, которые начинаются с bad

$appName$.exe -c "for file f from dir 'c:\dir' where f.name !~ '^bad.*' do md5;"


Поиск файла на диске C:\ по известному MD4 хэшу

$appName$.exe -c "for file f from dir 'c:\' where f.md4 == 'C58CDA49F00748A3BC0FCFA511D516CB' do find withsubs;"


Восстановление строки по её MD4 хэшу используя словарь по умолчанию

$appName$.exe -c "for string s from hash '3689CA24BF71B39B6612549D87DCEA68' do crack md4;"


Восстановление строки по её MD4 хэшу используя свой словарь

$appName$.exe -c "for string s from hash '3689CA24BF71B39B6612549D87DCEA68' let s.dict = '0123456789' do crack md4;"


Восстановление строки по её MD4 хэшу используя свой словарь сокращенная форма (одни цифры)

$appName$.exe -c "for string s from hash '3689CA24BF71B39B6612549D87DCEA68' let s.dict = '0-9' do crack md4;"


Восстановление строки по её MD4 хэшу используя свой словарь сокращенная форма (одни буквы в нижнем регистре)

$appName$.exe -c "for string s from hash '3689CA24BF71B39B6612549D87DCEA68' let s.dict = 'a-z' do crack md4;"


Восстановление строки по её MD4 хэшу используя свой словарь сокращенная форма (цифры, буквы в нижнем и верхнем регистрах)

$appName$.exe -c "for string s from hash '3689CA24BF71B39B6612549D87DCEA68' let s.dict = '0-9a-zA-Z' do crack md4;"


Восстановление строки по её MD4 хэшу используя свой словарь и определенные длины строки

$appName$.exe -c "for string s from hash '3689CA24BF71B39B6612549D87DCEA68' let s.dict = '0123456789', s.min = 2, s.max = 6 do crack md4;"


Использование файлов с запросами

Создайте текстовый файл queries.hq, например со следующим содержимым:

# string query - this is comment
for string '123' 
do sha1;

# file query  - this is comment
for file f 
from dir 'c:\dir' 
where f.name !~ '^bad.*' 
do md5;

Далее запустите компилятор указав ему на этот файл:

$appName$.exe queries.hq

Будет выполнено 2 запроса из файла


Использование переменных

Переменные удобны чтобы не дублировать одни и те же данные много раз при использовании их в разных запросах. Например:

# query that calculates md5 and sha1 for some files in a directory

let p = 'c:\dir';
let mask = '^bad.*';

for file f 
from dir p
where f.name !~ mask 
do md5;

for file f 
from dir p
where f.name !~ mask 
do sha1;
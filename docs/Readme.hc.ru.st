﻿$langName$ Readme 

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

  <string>                  hash algorithm. See docs for all possible values
  -f, --file=<file>         full path file to calculate hash sum for
  -d, --dir=<string>        full path to dir to calculate hash specified of all
                            content
  -e, --exclude=<string>    exclude files that match the pattern specified. It'
                            s possible to use several patterns separated by ;
  -i, --include=<string>    include only files that match the pattern specified.
                            It's possible to use several patterns separated by ;
  -s, --string=<string>     string to calculate hash sum for
  -m, --hash=<string>       hash to validate file or to find initial string (crack)
  -a, --dict=<string>       initial string's dictionary by default all digits,
                            upper and lower case latin symbols
  -n, --min=<int>           set minimum length of the string to restore using
                            option crack (c). 1 by default
  -x, --max=<int>           set maximum length of the string to restore  using
                            option crack (c). 10 by default
  -z, --limit=<number>      set the limit in bytes of the part of the file to
                            calculate hash for. The whole file by default will
                            be applied
  -q, --offset=<number>     set start position in the file to calculate hash
                            from zero by default
  -H, --search=<string>     hash to search a file that matches it
  -o, --save=<file>         save files' hashes into the file specified by full
                            path
  -r, --recursively         scan directory recursively
  -c, --crack               crack hash specified (find initial string) by option
                            --hash (-m)
  -p, --performance         test performance by cracking 12345 string hash
  -C, --command=<string>    query text from command line
  -F, --query=<file>        one or more query files
  -P, --param=<file>        path to file that will be validated using one or more
                            queries
  -S, --syntaxonly          only validate syntax. Do not run actions
  -t, --time                show calculation time (false by default)
  -l, --lower               output hash using low case (false by default)
      --sfv                 output hash in the SFV (Simple File Verification)
                            format (false by default)
  -h, --help                print this help and exit


Формальная грамматика языка

// Определения парсера

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

expr_string:
	STR source DO hash_clause
	;

expr_hash:
	STR id FROM HASH source let_clause? DO brute_force_clause
	;

expr_dir
	: FILE id FROM DIR source let_clause? where_clause? DO ( hash_clause WITHSUBS? | FIND WITHSUBS? )
	;

expr_file
	: FILE id FROM source (let_clause)? DO ( hash_clause | VALIDATE )
	;

expr_file_analyze
	: FILE id FROM PARAMETER where_clause DO VALIDATE
	;

source : STRING;
 
id : ID;

attr_clause : ID DOT attr ;

attr : str_attr | int_attr ;

hash_clause
    : ALG
    ;
    
brute_force_clause
	: CRACK hash_clause 
	;

let_clause
	: LET assign (COMMA assign)*
	;

where_clause
	: WHERE! boolean_expression
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
	: ID DOT 
	( relational_expr_str
	| relational_expr_int
	)
	;

relational_expr_str
	:	str_attr (EQUAL | NOTEQUAL | MATCH | NOTMATCH) (STRING | ID)
	;

relational_expr_int
	:	int_attr (EQUAL | NOTEQUAL | GE | LE | LEASSIGN | GEASSIGN) INT
	;

assign 
	: ID DOT ( str_attr ASSIGN_OP STRING | str_attr ASSIGN_OP ID | int_attr ASSIGN_OP INT )
	;
 
str_attr : NAME_ATTR | PATH_ATTR | DICT_ATTR | ALG ; 

int_attr : SIZE_ATTR | LIMIT_ATTR | OFFSET_ATTR | MIN_ATTR | MAX_ATTR ; 

// Определения лексера

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

COMMENT : ('#' | '/' '/') ~(EOL)* CR? (LF | EOF);

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
LEASSIGN : LE ASSIGN;
GEASSIGN : GE ASSIGN;


Примеры

Calculate SHA1 hash of string 123

$appName$.exe sha1 -s 123


Calculate SHA1 hash of a file

$appName$.exe sha1 -f file.txt


Calculate SHA1 hash of the part of the file (the first kilobyte)

$appName$.exe sha1 -f file.txt -z 1024


Calculate SHA1 hash of the part of the file (one kilobyte skiping the first 512 bytes)

$appName$.exe sha1 -f file.txt -z 1024 -q 512


Validate file using it's SHA1 hash

$appName$.exe sha1 -f file.txt -m 8BD7954C40C1E59A900F71EA3A266732609915B1


Calculate SHA1 of all files in c:\dir directory

$appName$.exe sha1 -d c:\dir


Calculate SHA1 of all files in c:\dir directory including all it's subdirectories

$appName$.exe sha1 -r -d c:\dir


Calculate SHA1 of all exe files in c:\dir directory

$appName$.exe sha1 -d c:\dir -i *.exe


Calculate SHA1 of all files in c:\dir directory excluding files with tmp extension

$appName$.exe sha1 -d c:\dir -e *.tmp


Calculate SHA1 of all exe and dll files in c:\dir directory

$appName$.exe sha1 -d c:\dir -i *.exe;*.dll


Calculate SHA1 of all exe files in c:\dir directory excluding files beginning with bad

$appName$.exe sha1 -d c:\dir -i *.exe -e bad*


Searching file on C:\ drive using known SHA1 hash

$appName$.exe sha1 -d c:\ -r -h 8BD7954C40C1E59A900F71EA3A266732609915B1


Restore string by it's SHA1 hash using default dictionary

$appName$.exe sha1 -с -m 40BD001563085FC35165329EA1FF5C5ECBDBBEEF


Restore string by it's SHA1 hash using user defined dictionary

$appName$.exe sha1 -с -m 40BD001563085FC35165329EA1FF5C5ECBDBBEEF -a 0123456789


Restore string by it's SHA1 hash using user defined dictionary short form (only digits)

$appName$.exe sha1 -с -m 40BD001563085FC35165329EA1FF5C5ECBDBBEEF -a 0-9


Restore string by it's SHA1 hash using user defined dictionary short form (low case latin letters)

$appName$.exe sha1 -с -m 40BD001563085FC35165329EA1FF5C5ECBDBBEEF -a a-z


Restore string by it's SHA1 hash using user defined dictionary short form (digits, upper and low case latin letters)

$appName$.exe sha1 -с -m 40BD001563085FC35165329EA1FF5C5ECBDBBEEF -a 0-9a-zA-Z


Restore string by it's SHA1 hash using user defined dictionary and string to restore min and max length

$appName$.exe sha1 -с -m 40BD001563085FC35165329EA1FF5C5ECBDBBEEF -a 0123456789 -n 2 -x 6


Test performance SHA1 using default dictionary

$appName$.exe sha1 -p


Test performance using user defined dictionary short form (only digits)

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
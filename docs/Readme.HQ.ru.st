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
- Поддержка переменных

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
-c [ --command ] <текст>  Текст запроса на языке $langName$
-p [ --param ] <путь>     Путь к файлу который будет проверен с помощью запроса
-s [ --syntaxonly ]       Проверить синтаксис запроса. Не запускать никаких действий
-t [ --time ]             Показывать время вычисления хэша (по умолчанию отключено)
-l [ --lower ]            Выводить хэш в нижнем регистре
-? [ --help ]             Показать помощь


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
    : MD5 | MD4 | SHA1 | SHA256 | SHA384 | SHA512 | CRC32 | WHIRLPOOL
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
 
str_attr : NAME_ATTR | PATH_ATTR | DICT_ATTR | MD5 | MD4 | SHA1 | SHA256 | SHA384 | SHA512 | CRC32 | WHIRLPOOL ; 

int_attr : SIZE_ATTR | LIMIT_ATTR | OFFSET_ATTR | MIN_ATTR | MAX_ATTR ; 

// Определения лексера

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

MD5: 'md5';	
SHA1: 'sha1' ;
SHA256: 'sha256' ;
SHA384: 'sha384' ;
SHA512: 'sha512' ;
MD4: 'md4' ;
CRC32: 'crc32' ;
WHIRLPOOL: 'whirlpool' ;

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

$appName$.exe -q "for file f from dir 'c:\dir' where f.name ~ '.*exe\$' do crc32;"


Вычисление MD5 хэша всех файлов каталога c:\dir кроме файлов с расширением tmp

$appName$.exe -q "for file f from dir 'c:\dir' where f.name !~ '.*tmp\$' do md5;"


Вычисление MD5 хэша всех exe и dll файлов каталога c:\dir

$appName$.exe -q "for file f from dir 'c:\dir' where f.name ~ '.*exe\$' or f.name ~ '.*dll\$' do md5;"


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

$appName$.exe -f queries.hq

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
$langName$ Readme 

What is $langName$? In short, it's declarative query language interpreter (or compiler, if you want) to calculate  hashes of strings and files. $langName$ features:

- string hash calculation
- file hash calculation, including only part file hash (defined by file part size and offset from the beginning)
- restoring original string by it's hash specified using brute force method (dictionary search)
- directory's files hash calculation with support of filtering files by size, name, path
- file validation using it's hash
- file searching using file hashes of the whole file or only the part of the file

Also there are:

- Brute force restoring time calculation
- Different case hash output (by default upper case)
- Queries from comand line and files support
- Comments support in queries' files
- Variables support

Following hashes supported:

- MD5
- MD4
- SHA1
- SHA256
- SHA384
- SHA512
- Whirlpool
- CRC32

Usage

$appName$ [OPTION] ...

Options:

-f [ --file ] <path>      Full path to $langName$ language queries file
-q [ --query ] <text>	  $langName$ language query text
-v [ --validate ]         Only validate syntax. Do not run actions
-t [ --time ]             Show hash calculation time (false by default)
-l [ --lower ]            output hash using low case (false by default)
-? [ --help ]             show help message


Formal language grammar

// Parser definitions

prog
	: statement+ | EOF
	;

     
statement
    :   expr NEWLINE
    | NEWLINE
    ;

expr:
	FOR (expr_string | expr_hash | expr_dir | expr_file) | expr_vardef
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

// Lexer definitions

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


Examples

Calculate SHA1 hash of string 123

$appName$.exe -q "for string '123' do sha1;"


Calculate MD5 hash of a file

$appName$.exe -q "for file f from 'file.txt' do md5;"


Calculate SHA384 hash of the part of the file (the first kilobyte)

$appName$.exe -q "for file f from 'file.txt' let f.limit = 1024 do sha384;"


Calculate SHA256 hash of the part of the file (one kilobyte skiping the first 512 bytes)

$appName$.exe -q "for file f from 'file.txt' let f.limit = 1024, f.offset = 512 do sha256;"


Validate file using it's MD4 hash

$appName$.exe -q "for file f from 'file.txt' let f.md4 = 'C58CDA49F00748A3BC0FCFA511D516CB' do validate;"


Calculate SHA512 of all files in c:\dir directory

$appName$.exe -q "for file f from dir 'c:\dir' do sha512;"


Calculate Whirlpool of all files in c:\dir directory including all it's subdirectories

$appName$.exe -q "for file f from dir 'c:\dir' do whirlpool withsubs;"


Calculate CRC32 of all exe files in c:\dir directory

$appName$.exe -q "for file f from dir 'c:\dir' where f.name ~ '.*exe\$' do crc32;"


Calculate MD5 of all files in c:\dir directory excluding files with tmp extension

$appName$.exe -q "for file f from dir 'c:\dir' where f.name !~ '.*tmp\$' do md5;"


Calculate MD5 of all exe and dll files in c:\dir directory

$appName$.exe -q "for file f from dir 'c:\dir' where f.name ~ '.*exe\$' or f.name ~ '.*dll\$' do md5;"


Calculate MD5 of all exe files in c:\dir directory excluding files beginning with bad

$appName$.exe -q "for file f from dir 'c:\dir' where f.name !~ '^bad.*' do md5;"


Searching file on C:\ drive using known MD4 hash

$appName$.exe -q "for file f from dir 'c:\' where f.md4 == 'C58CDA49F00748A3BC0FCFA511D516CB' do find withsubs;"


Restore string by it's MD4 hash using default dictionary

$appName$.exe -q "for string s from hash '3689CA24BF71B39B6612549D87DCEA68' do crack md4;"


Restore string by it's MD4 hash using user defined dictionary

$appName$.exe -q "for string s from hash '3689CA24BF71B39B6612549D87DCEA68' let s.dict = '0123456789' do crack md4;"


Restore string by it's MD4 hash using user defined dictionary short form (only digits)

$appName$.exe -q "for string s from hash '3689CA24BF71B39B6612549D87DCEA68' let s.dict = '0-9' do crack md4;"


Restore string by it's MD4 hash using user defined dictionary short form (low case latin letters)

$appName$.exe -q "for string s from hash '3689CA24BF71B39B6612549D87DCEA68' let s.dict = 'a-z' do crack md4;"


Restore string by it's MD4 hash using user defined dictionary short form (digits, upper and low case latin letters)

$appName$.exe -q "for string s from hash '3689CA24BF71B39B6612549D87DCEA68' let s.dict = '0-9a-zA-Z' do crack md4;"


Restore string by it's MD4 hash using user defined dictionary and string to restore min and max length

$appName$.exe -q "for string s from hash '3689CA24BF71B39B6612549D87DCEA68' let s.dict = '0123456789', s.min = 2, s.max = 6 do crack md4;"


Using queries' files

Create text file queries.hq with the following content for example:

# string query - this is comment
for string '123' 
do sha1;

# file query  - this is comment
for file f 
from dir 'c:\dir' 
where f.name !~ '^bad.*' 
do md5;

Then run compiler using this file:

$appName$.exe -f queries.hq

Two queries from the file will be executed


Variables usage

Variables are very useful so as not to duplicate same data in different queries. Example:

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

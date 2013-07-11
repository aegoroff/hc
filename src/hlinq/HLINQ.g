grammar HLINQ;

options {
    language = C;
    output=AST;
    ASTLabelType	= pANTLR3_BASE_TREE;
}

tokens
{
    ATTR_REF;
    HASH_STR;
    HASH_FILE;
    ANALYZE_FILE;
    HASH_DIR;
    BRUTE_FORCE;
    VAR_DEF;
    ALG_REF;
}

// While you can implement your own character streams and so on, they
// normally call things like LA() via funtion pointers. In general you will
// be using one of the pre-supplied input streams and you can instruct the
// generated code to access the input pointrs directly.
//
// For  8 bit inputs            : #define ANTLR3_INLINE_INPUT_ASCII
// For 16 bit UTF16/UCS2 inputs : #define ANTLR3_INLINE_INPUT_UTF16
//
// If your compiled recognizer might be given inputs from either of the sources
// or you have written your own character input stream, then do not define
// either of these.
//
@lexer::header
{
	#define	ANTLR3_INLINE_INPUT_ASCII
	#include "compiler.h"
#ifdef GTEST
  #include "displayError.h"
#endif
}

@parser::header {
	#define	MAX_STATEMENTS 10000
	#include "..\srclib\lib.h"
#ifdef GTEST
  #include "displayError.h"
#endif
}
 
@parser::apifuncs {
#ifdef GTEST
  RECOGNIZER->displayRecognitionError       = displayRecognitionErrorNew;
#endif
}

@lexer::apifuncs {
#ifdef GTEST
  RECOGNIZER->displayRecognitionError       = displayRecognitionErrorNew;
#endif
}

@members {
	long statementCount;
}

prog
@init {  statementCount = 0; }
	: statement+ | EOF!
	;

     
statement
    :   expr NEWLINE! 
    {
	    if (++statementCount > MAX_STATEMENTS) {
		    CrtPrintf("Too much statements. Max allowed \%i", MAX_STATEMENTS);
		    exit(1);
	    }
	}
    | NEWLINE!
    ;

expr:
	FOR (expr_string | expr_hash | expr_dir | expr_file | expr_file_analyze) | expr_vardef
    ;

expr_vardef:
	LET ID ASSIGN_OP STRING -> ^(VAR_DEF ID STRING)
	;

expr_string:
	STR source DO hash_clause -> ^(HASH_STR hash_clause source)
	;


expr_hash:
	STR id FROM HASH source let_clause? DO brute_force_clause -> ^(BRUTE_FORCE id brute_force_clause let_clause? source)
	;

expr_dir
	: FILE id FROM DIR source let_clause? where_clause? DO 
	( hash_clause WITHSUBS? -> ^(HASH_DIR hash_clause id let_clause? where_clause? WITHSUBS? source)
	| FIND WITHSUBS?        -> ^(HASH_DIR id let_clause? where_clause FIND WITHSUBS? source)
	)
	;

expr_file
	: FILE id FROM source (let_clause)? DO 
	( hash_clause -> ^(HASH_FILE hash_clause id let_clause? source) 
	| VALIDATE    -> ^(HASH_FILE id let_clause source) 
	)
	;

expr_file_analyze
	: FILE id FROM PARAMETER where_clause DO VALIDATE -> ^(ANALYZE_FILE id where_clause) 
	;

source : ID | STRING;

id : ID;

attr_clause : ID DOT attr -> ^(ATTR_REF ID attr) ;

attr : str_attr | int_attr ;

hash_clause
    : ALG -> ^(ALG_REF ALG)
    ;
    
brute_force_clause
	: CRACK hash_clause 
	;

let_clause
	: LET assign (COMMA assign)* -> assign+
	;

where_clause
	: WHERE! boolean_expression
    ;

boolean_expression
	: conditional_or_expression
	;

conditional_or_expression
	: conditional_and_expression (OR^ conditional_and_expression)*
	;

conditional_and_expression
	: not_expression (AND^ not_expression)* 
	;

not_expression
	: exclusive_or_expression
	| NOT_OP exclusive_or_expression -> ^(NOT_OP exclusive_or_expression)
	;

exclusive_or_expression
	:	relational_expr
	|	OPEN_BRACE boolean_expression CLOSE_BRACE -> boolean_expression
	;

relational_expr
	: ID DOT 
	( relational_expr_str -> ^(ATTR_REF ID relational_expr_str)
	| relational_expr_int -> ^(ATTR_REF ID relational_expr_int) 
	)
	;

relational_expr_str
	:	str_attr (EQUAL^ | NOTEQUAL^ | MATCH^ | NOTMATCH^) (STRING | ID)
	;

relational_expr_int
	:	int_attr (EQUAL^ | NOTEQUAL^ | GE^ | LE^ | LEASSIGN^ | GEASSIGN^) INT
	;

assign 
	: ID DOT 
	( str_attr ASSIGN_OP STRING -> ^(ATTR_REF ID ^(ASSIGN_OP str_attr STRING))
	| str_attr ASSIGN_OP ID -> ^(ATTR_REF ID ^(ASSIGN_OP str_attr ID))
	| int_attr ASSIGN_OP INT    -> ^(ATTR_REF ID ^(ASSIGN_OP int_attr INT))
	)
	;
 
str_attr : NAME_ATTR | PATH_ATTR | DICT_ATTR | ALG ; 

int_attr : SIZE_ATTR | LIMIT_ATTR | OFFSET_ATTR | MIN_ATTR | MAX_ATTR ; 

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
    | 'edonr224' 
    | 'edonr256' 
    | 'edonr384' 
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
WS  :   (' '|'\t'| EOL )+ { SKIP(); } ;
DOT	: '.' ;
COMMA: ',' ;	
OPEN_BRACE : '(';
CLOSE_BRACE : ')';

COMMENT : ('#' | '/' '/') ~(EOL)* CR? (LF | EOF) { SKIP(); } ;

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
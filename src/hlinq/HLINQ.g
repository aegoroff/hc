grammar HLINQ;

options {
    language = C;
    output=AST;
    ASTLabelType	= pANTLR3_BASE_TREE;
}

tokens
{
    ATTR_REF;
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
   #include "compiler.h"
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
	BOOL printCalcTime;
}

prog[apr_pool_t* root, BOOL onlyValidate, BOOL isPrintCalcTime]
	: statement+ | EOF!
	;

     
statement
    :   expr NEWLINE! | NEWLINE!
    ;

expr:
	FOR (expr_string | expr_hash | expr_dir | expr_file)
    ;

expr_string:
	STR s=STRING DO hash_clause
	;

expr_hash:
	STR id FROM HASH s=STRING (let_clause)? DO brute_force_clause
	;

expr_dir
	: FILE id FROM DIR s=STRING (let_clause)? (where_clause)? DO (hash_clause | find_clause) (recursively)?
	;

expr_file
	: FILE id FROM s=STRING (let_clause)? DO hash_clause
	;
    
id : ID;

attr_clause : ID DOT attr -> ^(ATTR_REF ID attr) ;

attr : str_attr | int_attr ;

find_clause:
    'find'
    ;

hash_clause
    : MD5 | MD4 | SHA1 | SHA256 | SHA384 | SHA512 | CRC32 | WHIRLPOOL
    ;
    
brute_force_clause
	: 'crack' hash_clause 
	;

recursively
	: 'recursively'	
	;

let_clause
	: LET assign (COMMA assign)* -> assign+
	;

where_clause
	: 'where' boolean_expression
    ;

boolean_expression
	: conditional_or_expression
	;

conditional_or_expression
	: conditional_and_expression (OR^ conditional_and_expression)*
	;

conditional_and_expression
	: exclusive_or_expression (AND^ exclusive_or_expression)* 
	;

exclusive_or_expression
	:	relational_expr
	|	OPEN_BRACE! boolean_expression CLOSE_BRACE!
	;

relational_expr
	: ID DOT relational_expr_str -> ^(ATTR_REF ID relational_expr_str)
	| ID DOT relational_expr_int -> ^(ATTR_REF ID relational_expr_int)
	;

relational_expr_str
	:	str_attr (EQUAL^ | NOTEQUAL^ | MATCH^ | NOTMATCH^) STRING
	;

relational_expr_int
	:	int_attr (EQUAL^ | NOTEQUAL^ | GE^ | LE^ | LEASSIGN^ | GEASSIGN^) INT
	;

assign 
	:	ID DOT str_attr ASSIGN_OP STRING -> ^(ATTR_REF ID ^(ASSIGN_OP str_attr STRING))
	|	ID DOT int_attr ASSIGN_OP INT -> ^(ATTR_REF ID ^(ASSIGN_OP int_attr INT))
	;
 
str_attr : 'name' | 'path' | 'dict' | hash_clause ; 

int_attr : 'size' | 'limit' | 'offset' | 'min' | 'max'; 

OR: 'or' ;

AND: 'and' ;

FOR: 'for' ;

FROM: 'from' ;

DO: 'do' ;

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


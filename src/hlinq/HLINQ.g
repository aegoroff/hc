grammar HLINQ;

options {
    language = C;
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

}

prog[apr_pool_t* root, BOOL onlyValidate]
@init { InitProgram($onlyValidate, $root); }
	: statement+ | EOF
	;

     
statement
@init {
	OpenStatement(); 
}
@after {
	CloseStatement(RECOGNIZER->state->errorCount);
}
    :   expr NEWLINE | NEWLINE
    ;

expr:
	FOR (expr_string | expr_hash | expr_dir | expr_file)
    ;

expr_string:
	'string' {  RegisterIdentifier("_s_", CtxTypeString); } FROM s=STRING { SetSource($s.text->chars); } DO hash_clause
	;

expr_hash:
	'hash' id[CtxTypeHash] FROM s=STRING { SetSource($s.text->chars); } (let_clause)? DO brute_force_clause
	;

expr_dir:
	'dir' id[CtxTypeDir] FROM s=STRING { SetSource($s.text->chars); } (let_clause)? (where_clause)? DO (hash_clause | find_clause)
	;

expr_file:
	'file' id[CtxTypeFile] FROM s=STRING { SetSource($s.text->chars); } (let_clause)? DO hash_clause
	;
    
id[CtxType contextType]
	: ID
	{
		 RegisterIdentifier($ID.text->chars, $contextType);
	};

attr_clause: id_ref DOT attr ;

attr:
    ( str_attr | int_attr )
    ;

find_clause:
    'find'
    ;

hash_clause:
    (md5 | md4 | sha1 | sha256 | sha384 | sha512 | crc32 | whirlpool)
    ;
  
md5	:	MD5 {  SetHashAlgorithm(AlgMd5); };
md4	:	MD4 {  SetHashAlgorithm(AlgMd4); };
sha1	:	SHA1 {  SetHashAlgorithm(AlgSha1); };
sha256	:	SHA256 {  SetHashAlgorithm(AlgSha256); };
sha384	:	SHA384 {  SetHashAlgorithm(AlgSha384); };
sha512	:	SHA512 {  SetHashAlgorithm(AlgSha512); };
crc32	:	CRC32 {  SetHashAlgorithm(AlgCrc32); };
whirlpool	:	WHIRLPOOL {  SetHashAlgorithm(AlgWhirlpool); };
    
brute_force_clause
	:	'crack' hash_clause 
	{ 
		SetBruteForce();
	}
	;

let_clause:
	LET assign (COMMA assign)*
	;

where_clause:
    'where' boolean_expression
    ;

boolean_expression:
	conditional_or_expression;

conditional_or_expression:
	conditional_and_expression  (OR conditional_and_expression)* ;

conditional_and_expression:
	exclusive_or_expression   (AND exclusive_or_expression)* ;

exclusive_or_expression:
	id_ref DOT
	(
		(sa=str_attr (COND_OP | COND_OP_STR) s=STRING { AssignStrAttribute($sa.code, $s.text->chars); })
		| 
		(ia=int_attr (COND_OP | COND_OP_INT) i=INT { AssignIntAttribute($ia.code, $i.text->chars); })
	)
	|
	OPEN_BRACE boolean_expression CLOSE_BRACE
	;

assign :
	id_ref DOT (
		(sa=str_attr ASSIGN_OP s=STRING { AssignStrAttribute($sa.code, $s.text->chars); })
		| 
		(ia=int_attr ASSIGN_OP i=INT { AssignIntAttribute($ia.code, $i.text->chars); })
	)
	;
 
id_ref
	: ID
	{
		if (!CallAttiribute($ID.text->chars)) {
			RECOGNIZER->state->exception = antlr3ExceptionNew(ANTLR3_RECOGNITION_EXCEPTION, "unknown identifier", "error: unknown identifier", ANTLR3_FALSE);
			RECOGNIZER->state->exception->token = $ID;
			RECOGNIZER->state->error = ANTLR3_RECOGNITION_EXCEPTION;
		};
	}
;
 
str_attr returns[int code]
@init { $code = -1; }
:
    (
    'name' { $code = 0; } | 
    'path' { $code = 1; } | 
    'dict' { $code = 2; } | 
     MD5 { $code = 3; } | 
     SHA1 { $code = 4; } | 
     SHA256 { $code = 5; } | 
     SHA384 { $code = 6; } | 
     SHA512 { $code = 7; } | 
     MD4 { $code = 8; } | 
     CRC32 { $code = 9; } | 
     WHIRLPOOL { $code = 10; } 
     )
    ; 

int_attr returns[int code]
@init { $code = -1; }
:
    ('size' { $code = 0; } | 'limit' { $code = 1; } | 'offset' { $code = 2; } | 'min' { $code = 3; } | 'max' { $code = 4; } )
    ; 

OR: 'or' ;

AND: 'and' ;

FOR: 'for' ;

FROM: 'from' ;

DO: 'do' ;

LET	: 'let' ;


MD5: 'md5';	
SHA1: 'sha1' ;
SHA256: 'sha256' ;
SHA384: 'sha384' ;
SHA512: 'sha512' ;
MD4: 'md4' ;
CRC32: 'crc32' ;
WHIRLPOOL: 'whirlpool' ;

fragment
STRING1
    : '\'' ( options {greedy=false;} : ~('\u0027' | '\u000A' | '\u000D'))* '\''
    ;

fragment
STRING2
    : '"'  ( options {greedy=false;} : ~('\u0022' | '\u000A' | '\u000D'))* '"'
    ;

STRING
    : STRING1 | STRING2
    ;

ID:
    ID_START ID_PART* ;

fragment
ID_START
	: '_' | 'A'..'Z' | 'a'..'z' ;
fragment
ID_PART
: ID_START | '0'..'9' ;

INT :   '0'..'9'+ ;
ASSIGN_OP : ASSIGN;
COND_OP :   EQUAL | NOT ASSIGN;
COND_OP_STR : MATCH | NOT MATCH;
COND_OP_INT : GE | LE | LE ASSIGN | GE ASSIGN;
NEWLINE: ';';
WS  :   (' '|'\t'| EOL )+ { SKIP(); } ;
DOT	: '.' ;
COMMA: ',' ;	
OPEN_BRACE
	:	'(';
CLOSE_BRACE
	:	')';

COMMENT 
    : ('#' | '/' '/') ~(EOL)* CR? (LF | EOF) { SKIP(); }
    ;

fragment
EOL
    : LF | CR
    ;

fragment
LF 
	:	'\n' ;

fragment
CR 
	:	'\r' ;
 
PLUS:	'+' ;

fragment
ASSIGN:	'=' ;

fragment
EQUAL:	ASSIGN ASSIGN ;

fragment
NOT:	'!' ;

fragment
GE:	'>' ;

fragment
LE:	'<' ;

fragment
MATCH:	'~' ;


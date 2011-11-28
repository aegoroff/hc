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
	BOOL printCalcTime;
}

prog[apr_pool_t* root, BOOL onlyValidate, BOOL isPrintCalcTime]
@init { 
	printCalcTime = $isPrintCalcTime;
	InitProgram($onlyValidate, $root); 
}
	: statement+ | EOF
	;

     
statement
@init {
	OpenStatement(); 
}
@after {
	CloseStatement(RECOGNIZER->state->errorCount, printCalcTime);
}
    :   expr NEWLINE | NEWLINE
    ;

expr:
	FOR (expr_string | expr_hash | expr_dir | expr_file)
    ;

expr_string:
	STR {  RegisterIdentifier("_s_", CtxTypeString); } s=STRING { SetSource($s.text->chars); } DO hash_clause
	;

expr_hash:
	STR id[CtxTypeHash] FROM HASH s=STRING { SetSource($s.text->chars); } (let_clause)? DO brute_force_clause
	;

expr_dir:
	FILE id[CtxTypeDir] FROM DIR s=STRING { SetSource($s.text->chars); } (let_clause)? (where_clause)? DO (hash_clause | find_clause) (recursively)?
	;

expr_file:
	FILE id[CtxTypeFile] FROM s=STRING { SetSource($s.text->chars); } (let_clause)? DO hash_clause
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

recursively
	: 'recursively'	
	{
		SetRecursively();
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
		(sa=str_attr c=cond_op_str s=STRING { WhereClauseCallString($sa.code, $s.text->chars, $c.opcode); })
		| 
		(ia=int_attr c=cond_op_int i=INT { WhereClauseCallInt($ia.code, $i.text->chars, $c.opcode); })
	)
	|
	OPEN_BRACE boolean_expression CLOSE_BRACE
	;

	
cond_op returns [CondOp opcode] 
@init {
	opcode = CondOpUndefined; 
}
: EQUAL {$opcode = CondOpEq;} | NOTEQUAL {$opcode = CondOpNotEq;} ;

cond_op_str returns [CondOp opcode] 
@init {
	opcode = CondOpUndefined; 
}
: 
	MATCH {$opcode = CondOpMatch;} | 
	NOT MATCH {$opcode = CondOpNotMatch;} | 
	op=cond_op {$opcode = $op.opcode;};

cond_op_int returns [CondOp opcode] 
@init {
	opcode = CondOpUndefined; 
}
:
	GE {$opcode = CondOpGe;} | 
	LE {$opcode = CondOpLe;} | 
	LE ASSIGN {$opcode = CondOpLeEq;} | 
	GE ASSIGN {$opcode = CondOpGeEq;} | 
	op=cond_op {$opcode = $op.opcode;};

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
@init { $code = StrAttrUndefined; }
:
    (
    'name' { $code = StrAttrName; } | 
    'path' { $code = StrAttrPath; } | 
    'dict' { $code = StrAttrDict; } | 
     MD5 { $code = StrAttrMd5; } | 
     SHA1 { $code = StrAttrSha1; } | 
     SHA256 { $code = StrAttrSha256; } | 
     SHA384 { $code = StrAttrSha384; } | 
     SHA512 { $code = StrAttrSha512; } | 
     MD4 { $code = StrAttrMd4; } | 
     CRC32 { $code = StrAttrCrc32; } | 
     WHIRLPOOL { $code = StrAttrWhirlpool; } 
     )
    ; 

int_attr returns[IntAttr code]
@init { $code = IntAttrUndefined; }
:
    ('size' { $code = IntAttrSize; } | 'limit' { $code = IntAttrLimit; } | 'offset' { $code = IntAttrOffset; } | 'min' { $code = IntAttrMin; } | 'max' { $code = IntAttrMax; } )
    ; 

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

EQUAL:	ASSIGN ASSIGN ;
NOTEQUAL:	NOT ASSIGN ;

fragment
ASSIGN:	'=' ;

fragment
NOT:	'!' ;

GE:	'>' ;
LE:	'<' ;
MATCH:	'~' ;


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
	#define	MAX_STATEMENTS 10000
	#include "..\srclib\lib.h"
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
	long statementCount;
}

prog[apr_pool_t* root, ProgramOptions* po, const char* param]
@init {  
	InitProgram($po, $param, $root); 
	statementCount = 0; 
}
	: statement+ | EOF
	;

     
statement
@init {
        OpenStatement(RECOGNIZER->state); 
}
@after {
        CloseStatement();
}
    :   expr NEWLINE
    | NEWLINE
    ;

expr:
	FOR (expr_string | expr_hash | expr_dir | expr_file | expr_file_analyze) | expr_vardef
    ;

expr_vardef:
	LET ID ASSIGN_OP s=STRING { RegisterVariable($ID.text->chars, $s.text->chars); }
	;

expr_string
@init {
  RegisterIdentifier((pANTLR3_UINT8)"_s_"); 
}
	: STR source DO hash_clause
	;


expr_hash:
	STR id FROM HASH source let_clause? DO brute_force_clause
	;

expr_dir
	: FILE id FROM DIR source let_clause? where_clause? DO 
	( hash_clause (WITHSUBS { SetRecursively(); })?
	| FIND { SetFindFiles(); } (WITHSUBS { SetRecursively(); })?
	)
	;

expr_file
	: FILE id FROM source (let_clause)? DO  ( hash_clause | VALIDATE )
	;

expr_file_analyze
	: FILE id FROM PARAMETER where_clause DO VALIDATE
	;

source : ID { SetSource($ID.text->chars, $ID); }
	| s=STRING { SetSource($s.text->chars, NULL); }
	;

id : ID;

attr_clause : ID DOT attr ;

attr : str_attr | int_attr ;

hash_clause
    : ALG {  SetHashAlgorithmIntoContext($ALG.text->chars);  }
    ;
    
brute_force_clause
	: CRACK hash_clause { SetBruteForce(); }
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
	| NOT_OP exclusive_or_expression { WhereClauseCond(CondOpNot, $NOT_OP); }
	;

exclusive_or_expression
	:	relational_expr
	|	OPEN_BRACE boolean_expression CLOSE_BRACE
	;

relational_expr
	: ID DOT ( relational_expr_str | relational_expr_int ) { CallAttiribute($ID.text->chars, $ID); }
	;

relational_expr_str
	:	l=str_attr 
	( o=EQUAL r=STRING { WhereClauseCall($l.code, $r.text->chars, CondOpEq, $o, $l.text->chars); }
	| o=EQUAL r=ID { WhereClauseCall($l.code, GetValue($r.text->chars, $r), CondOpEq, $o, $l.text->chars); }
	| o=NOTEQUAL r=STRING { WhereClauseCall($l.code, $r.text->chars, CondOpNotEq, $o, $l.text->chars); }
	| o=NOTEQUAL r=ID { WhereClauseCall($l.code, GetValue($r.text->chars, $r), CondOpNotEq, $o, $l.text->chars); }
	| o=MATCH r=STRING { WhereClauseCall($l.code, $r.text->chars, CondOpMatch, $o, $l.text->chars); }
	| o=MATCH r=ID { WhereClauseCall($l.code, GetValue($r.text->chars, $r), CondOpMatch, $o, $l.text->chars); }
	| o=NOTMATCH r=STRING { WhereClauseCall($l.code, $r.text->chars, CondOpNotMatch, $o, $l.text->chars); }
	| o=NOTMATCH r=ID { WhereClauseCall($l.code, GetValue($r.text->chars, $r), CondOpNotMatch, $o, $l.text->chars); }
	)
	;

relational_expr_int
	:	l=int_attr 
	(
	EQUAL r=INT { WhereClauseCall($l.code, $r.text->chars, CondOpEq, $EQUAL, $l.text->chars); }
	| NOTEQUAL r=INT { WhereClauseCall($l.code, $r.text->chars, CondOpNotEq, $NOTEQUAL, $l.text->chars); }
	| GE r=INT { WhereClauseCall($l.code, $r.text->chars, CondOpGe, $GE, $l.text->chars); }
	| LE r=INT { WhereClauseCall($l.code, $r.text->chars, CondOpLe, $LE, $l.text->chars); }
	| LEASSIGN r=INT { WhereClauseCall($l.code, $r.text->chars, CondOpLeEq, $LEASSIGN, $l.text->chars); }
	| GEASSIGN r=INT { WhereClauseCall($l.code, $r.text->chars, CondOpGeEq, $GEASSIGN, $l.text->chars); }
	)
	;

assign 
	: left=ID DOT 
	( sa=str_attr ASSIGN_OP s=STRING 
	{ 
		if(CallAttiribute($left.text->chars, $left)) {
			AssignAttribute($sa.code, $s.text->chars, $s, $sa.name);
		}
	}
	| sa=str_attr ASSIGN_OP right=ID
	{ 
		if(CallAttiribute($left.text->chars, $left)) {
			AssignAttribute($sa.code, GetValue($right.text->chars, $right), $right, $sa.name);
		}
	}
	| ia=int_attr ASSIGN_OP i=INT
	{ 
		if(CallAttiribute($left.text->chars, $left)){
			AssignAttribute($ia.code, $i.text->chars, $i, NULL);
		}
	}
	)
	;
 
str_attr returns[Attr code, pANTLR3_UINT8 name] 
@init { 
    $code = AttrUndefined; 
    $name = NULL;
}
	: NAME_ATTR { $code = AttrName; }
	| PATH_ATTR { $code = AttrPath; }
	| DICT_ATTR { $code = AttrDict; }
	| ALG { $code = AttrHash; $name = $ALG.text->chars;  }
	; 

int_attr returns[Attr code]
@init { $code = AttrUndefined; }
	: SIZE_ATTR { $code = AttrSize; } 
	| LIMIT_ATTR { $code = AttrLimit; } 
	| OFFSET_ATTR { $code = AttrOffset; } 
	| MIN_ATTR { $code = AttrMin; } 
	| MAX_ATTR { $code = AttrMax; } 
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
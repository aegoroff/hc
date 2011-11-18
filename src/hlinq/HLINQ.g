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
}

@parser::header {
   #include "compiler.h"
}
 
@members {

}

prog[apr_pool_t* root, BOOL onlyValidate]
@init { InitProgram($onlyValidate, $root); }
	: statement+ 
	;

     
statement
scope {
	const char* id
}
@init {
	OpenStatement(); 
}
@after {
	CloseStatement($statement::id);
}
    :   expr NEWLINE |   NEWLINE
    ;

expr:
    FOR id 'in' searchIn=STRING (recursively)? (let_clause)? (where_clause)? do_clause_file
    {
		SetSearchRoot($searchIn.text->chars, $statement::id);
	}
	|
	FOR s=STRING ('as' id let_clause)? do_clause_string 
    ;
    
id
	: ID
	{
		$statement::id = (const char*)$ID.text->chars;
		CreateStatementContext($statement::id);
	};

recursively:
    'recursively'
    {
    	SetRecursively($statement::id);
    }
    ;
    
do_clause_file:
    DO (print_clause | delete_clause | copy_clause | move_clause | hash_clause);
    
do_clause_string:
	DO (hash_clause| brute_force_clause);
    
print_clause:
    'print' print (PLUS print)*
    ;
    
print:
	attr_clause | STRING | INT
	;

attr_clause:
    ID DOT attr
    {
		if (!CallAttiribute($ID.text->chars)) {
			// TODO: implement error
		};
	}
    ;

attr:
    ( str_attr | int_attr )
    ;

delete_clause:
    'delete'
    ;

copy_clause:
    'copy' s=STRING
    {
		SetActionTarget($s.text->chars, $statement::id);
	};

move_clause:
    'move' s=STRING
    {
		SetActionTarget($s.text->chars, $statement::id);
	};

hash_clause:
    HASH
    ;
    
brute_force_clause
	:	'crack' HASH
	;

let_clause:
	LET exclusive_or_expression (COMMA exclusive_or_expression)*
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
	ID DOT ((str_attr (COND_OP | COND_OP_STR) STRING) | (int_attr (COND_OP | COND_OP_INT) INT))
	{
		if (!CallAttiribute($ID.text->chars)) {
			// TODO: implement error
		};
	}
	;
 
str_attr:
    ('name' | 'path' | 'dict' | HASH )
    ; 

int_attr:
    ('size' | 'limit' | 'offset' | 'min' | 'max' )
    ; 

OR: 'or' ;

AND: 'and' ;

FOR: 'for' ;

DO: 'do' ;

LET	: 'let' ;

HASH:
    ('md5' | 'sha1' | 'sha256' | 'sha384' | 'sha512' | 'crc32' | 'whirlpool')
    ;

fragment
STRING1
    : '\'' ( options {greedy=false;} : ~('\u0027' | '\u005C' | '\u000A' | '\u000D') | ECHAR )* '\''
    ;

fragment
STRING2
    : '"'  ( options {greedy=false;} : ~('\u0022' | '\u005C' | '\u000A' | '\u000D') | ECHAR )* '"'
    ;

STRING
    : STRING1 | STRING2
    ;

fragment
ECHAR
    : '\\' ('t' | 'b' | 'n' | 'r' | 'f' | '\\' | '"' | '\'')
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
COND_OP :   EQUAL | NOT EQUAL;
COND_OP_STR : MATCH | NOT MATCH;
COND_OP_INT : GE | LE | LE EQUAL | GE EQUAL;
NEWLINE: ';';
WS  :   (' '|'\t'| EOL )+ { SKIP(); } ;
DOT	: '.' ;
COMMA: ',' ;	

COMMENT 
    : ('#' | '/' '/') ( options{greedy=false;} : .)* EOL { SKIP(); }
    ;

fragment
EOL
    : '\n' | '\r'
    ;
    
PLUS:	'+' ;

fragment
EQUAL:	'=' ;

fragment
NOT:	'!' ;

fragment
GE:	'>' ;

fragment
LE:	'<' ;

fragment
MATCH:	'~' ;


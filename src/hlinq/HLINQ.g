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

prog[apr_pool_t* root]
@init { InitProgram($root); }
	: statement+ 
	;

     
statement
scope {
	const char* identifier
}
@init { OpenStatement(); }
@after { CloseStatement($statement::identifier); }
    :   expr NEWLINE |   NEWLINE
    ;

expr:
    'for' identifier 'in' searchIn=STRING_LITERAL ('recursively')? (whereClause)? doClause
    {
		SetSearchRoot($searchIn.text->chars, $statement::identifier);
	}
    ;
    
identifier
	: IDENTIFIER
	{
		$statement::identifier = (const char*)$IDENTIFIER.text->chars;
		CreateStatementContext($statement::identifier);
	};
    
doClause:
    'do' (printClause | deleteClause | copyClause | moveClause | HASH);
    
printClause:
    'print' printItem (PLUS printItem)*
    ;
    
printItem:
	attrCall | STRING_LITERAL | INT
	;

deleteClause:
    'delete'
    ;

copyClause:
    'copy' s=STRING_LITERAL
    {
		SetActionTarget($s.text->chars, $statement::identifier);
	};

moveClause:
    'move' s=STRING_LITERAL
    {
		SetActionTarget($s.text->chars, $statement::identifier);
	};
    
 
whereClause:
    'where' boolean_expression
    ;

boolean_expression:
	conditional_or_expression;


exclusive_or_expression:
	attrCall COND_OPERATOR  ( STRING_LITERAL | INT )
	;
conditional_and_expression:
	exclusive_or_expression   ('and'   exclusive_or_expression)* ;
conditional_or_expression:
	conditional_and_expression  ('or'   conditional_and_expression)* ;

attrCall:
    IDENTIFIER '.' attrClause 
    {
    	CallAttiribute($IDENTIFIER.text->chars);
    }
    ;

attrClause:
    ('name' | 'size' | 'limit' | 'offset' | HASH )
    ; 

HASH:
    ('md5' | 'sha1' | 'sha256' | 'sha384' | 'sha512' | 'crc32' | 'whirlpool')
    ;

fragment
STRING_LITERAL1
    : '\'' ( options {greedy=false;} : ~('\u0027' | '\u005C' | '\u000A' | '\u000D') | ECHAR )* '\''
    ;

fragment
STRING_LITERAL2
    : '"'  ( options {greedy=false;} : ~('\u0022' | '\u005C' | '\u000A' | '\u000D') | ECHAR )* '"'
    ;

STRING_LITERAL
    : STRING_LITERAL1 | STRING_LITERAL2
    ;

fragment
ECHAR
    : '\\' ('t' | 'b' | 'n' | 'r' | 'f' | '\\' | '"' | '\'')
    ;

IDENTIFIER:
    IdentifierStart IdentifierPart* ;

fragment
IdentifierStart
	: '_' | 'A'..'Z' | 'a'..'z' ;
fragment
IdentifierPart
: IdentifierStart | '0'..'9' ;

INT :   '0'..'9'+ ;
COND_OPERATOR :   EQUAL | GE | LE | LE EQUAL | GE EQUAL | NOT EQUAL | MATCH | NOT MATCH ;
NEWLINE: ';';
WS  :   (' '|'\t'| EOL )+ { SKIP(); } ;

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


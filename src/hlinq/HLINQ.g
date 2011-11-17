grammar HLINQ;

options {
    language = C;
    backtrack=true;
}

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
@init { OpenStatement(); }
    :   expr NEWLINE 
    {
		CloseStatement();
    }
    |   NEWLINE
    {
		CloseStatement();
    }
    ;

expr:
    'for' identifierSet 'in' setSearchRoot ('recursively')? (whereClause)? doClause 
    ;
    
identifierSet
	: IDENTIFIER
	{
		RegisterIdentifier($IDENTIFIER.text->chars);
	};

setSearchRoot
	: s=STRING_LITERAL
	{
		SetSearchRoot($s.text->chars);
	};
    
doClause:
    'do' (printClause | deleteClause | copyClause | moveClause | HASH);
    
printClause:
    'print' attrCall ( PLUS STRING_LITERAL | PLUS STRING_LITERAL PLUS attrCall )*
    ;

deleteClause:
    'delete'
    ;

copyClause:
    'copy' s=STRING_LITERAL
    {
		SetActionTarget($s.text->chars);
	};

moveClause:
    'move' s=STRING_LITERAL
    {
		SetActionTarget($s.text->chars);
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
    {
      SetCurrentString($text->chars);
    }
    ;

fragment
STRING_LITERAL2
    : '"'  ( options {greedy=false;} : ~('\u0022' | '\u005C' | '\u000A' | '\u000D') | ECHAR )* '"'
    {
      SetCurrentString($text->chars);
    }
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


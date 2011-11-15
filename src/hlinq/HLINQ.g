grammar HLINQ;

options {
    language = C;
}

@lexer::header
{
#define	ANTLR3_INLINE_INPUT_ASCII
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
    |   NEWLINE
    {
	CloseStatement();
    }
    ;

expr:
    'for' identifierSet 'in' STRING_LITERAL ('recursively')? (whereClause)? doClause 
    ;
    
identifierSet
	: IDENTIFIER
	{
           RegisterIdentifier($IDENTIFIER.text->chars);
        };
    
doClause:
    'do' (printClause | deleteClause | copyClause | moveClause | hashClause);
    
printClause:
    'print' attrCall ( '+' STRING_LITERAL | '+' STRING_LITERAL '+' attrCall )*
    ;

deleteClause:
    'delete'
    ;

copyClause:
    'copy' STRING_LITERAL
    ;

moveClause:
    'move' STRING_LITERAL
    ;
    
hashClause:
    ('md5' | 'sha1' | 'sha256' | 'sha384' | 'sha512' | 'crc32' | 'whirlpool')
    ;

attrClause:
    ('name' | 'size' | 'limit' | 'offset' | 'hash' )
    ; 
 
whereClause:
    'where' boolean_expression
    ;

boolean_expression:
	conditional_or_expression;


exclusive_or_expression:
	attrCall COND_OPERATOR  ( STRING_LITERAL | INT )
	| 
	attrCall hashClause STRING_LITERAL
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

fragment
STRING_LITERAL1
    : '\'' ( options {greedy=false;} : ~('\u0027' | '\u005C' | '\u000A' | '\u000D') | ECHAR )* '\''
    {
      //Text = Text.Substring(1, Text.Length - 2);
    }
    ;

fragment
STRING_LITERAL2
    : '"'  ( options {greedy=false;} : ~('\u0022' | '\u005C' | '\u000A' | '\u000D') | ECHAR )* '"'
    {
      //Text = Text.Substring(1, Text.Length - 2);
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
COND_OPERATOR :   '=' | '>' | '<' | '<=' | '>=' | '!=' | '~' | '!~' ;
NEWLINE: ';';
WS  :   (' '|'\t'| EOL )+ {$channel=HIDDEN;} ;

COMMENT 
    : ('#' | '/' '/') ( options{greedy=false;} : .)* EOL { $channel=HIDDEN; }
    ;

fragment
EOL
    : '\n' | '\r'
    ;


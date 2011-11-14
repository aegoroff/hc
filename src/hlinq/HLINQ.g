grammar HLINQ;

options {
    language = C;
    ASTLabelType=pANTLR3_BASE_TREE;
}

@lexer::header
{
#define	ANTLR3_INLINE_INPUT_ASCII
}

@parser::header {

}
 
@members {

}

prog: statement+ ;

     
statement:   expr NEWLINE 
    |   NEWLINE
    ;

expr:
    'for' ID 'in' STRING_LITERAL ('recursively')? ('where' whereClause)? doClause;
    
doClause:
    'do' ('print' printClause | deleteClause | 'copy' STRING_LITERAL | 'move' STRING_LITERAL );
    
printClause:
    attrCall ( '+' STRING_LITERAL | '+' STRING_LITERAL '+' attrCall )*
    ;

deleteClause:
    'delete' ID
    ;
 
whereClause:
    HASH '=' STRING_LITERAL
    ;

attrCall:
    ID '.' ID
    ;

STRING_LITERAL
    :  '"' STRING_GUTS '"'
    ;

fragment
STRING_GUTS :	( EscapeSequence | ~('\\'|'"') )* ;

fragment
EscapeSequence
    :   '\\' ('b'|'t'|'n'|'f'|'r'|'\"'|'\''|'\\')
    |   OctalEscape
    ;
    
fragment
OctalEscape
    :   '\\' ('0'..'3') ('0'..'7') ('0'..'7')
    |   '\\' ('0'..'7') ('0'..'7')
    |   '\\' ('0'..'7')
    ;

ID  :   ('a'..'z'|'A'..'Z')+ ;
HASH  :   ('md5'|'MD5'|'sha1'|'SHA1') ;
INT :   '0'..'9'+ ;
NEWLINE: ';' ;
WS  :   (' '|'\t'|'\n'|'\r')+ {$channel=HIDDEN;} ;

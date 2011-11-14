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
    |   ID '=' expr NEWLINE
    |   NEWLINE
    ;

expr returns [int value]
    :   e=multExpr {$value = $e.value;}
        (   
        '+' e=multExpr {$value += $e.value;}
        |   
        '-' e=multExpr {$value -= $e.value;}
        )*
    ;
 
multExpr returns [int value]
    :   e=atom {$value = $e.value;}
    	(
    	'*' e=atom {$value *= $e.value;}
    	|
    	'/' e=atom {$value /= $e.value;}
    	)*
    ; 

 

atom returns [int value]
@init {
	$value=0; // init return value
}
    :   // value of an INT is the int computed from char sequence
        INT


    |   ID // variable reference


        // value of parenthesized expression is just the expr value
    |   '(' expr ')' {$value = $expr.value;}
    ;


ID  :   ('a'..'z'|'A'..'Z')+ ;
INT :   '0'..'9'+ ;
NEWLINE: '\r'? '\n' | ';' ;
WS  :   (' '|'\t'|'\n'|'\r')+ {$channel=HIDDEN;} ;

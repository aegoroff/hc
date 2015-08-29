%glr-parser
%expect 1

%{
    #include <stdio.h>
    #include <stdlib.h>
	extern int yylineno;
    extern char *yytext;
	int yyerror(char *s);
	int yylex();
%}

%code requires
{
	#include "lib.h"
	#include "frontend.h"
}


%union {
	CondOp_t RelOp;
	Ordering_t Ordering;
	long long Number;
	char* String;
	TypeInfo_t* Type;
	fend_node_t* Node;
}

%start translation_unit

%token SEMICOLON
%token FROM
%token <Type> TYPE
%token LET
%token ASSIGN
%token WHERE
%token ON
%token EQUALS
%token JOIN
%token ORDERBY
%token COMMA
%token <Ordering> ASCENDING
%token <Ordering> DESCENDING
%token SELECT
%token GRP
%token BY
%token OPEN_PAREN
%token CLOSE_PAREN
%token OPEN_BRACE
%token CLOSE_BRACE
%token <String> IDENTIFIER

%token <Number> INTEGER
%token <String> STRING
%token DOT

/* operators and precedence levels */

%nonassoc INTO
%left OR
%left AND
%nonassoc WITHIN
%left NOT
%left <RelOp> REL_OP

%type <Type> type
%type <Node> typedef
%type <Node> attribute
%type <Node> query_expression
%type <Node> expression
%type <Node> unary_expression
%type <Node> relational_expr
%type <Node> boolean_expression
%type <Node> conditional_or_expression
%type <Node> conditional_and_expression
%type <Node> not_expression
%type <Node> exclusive_or_expression
%type <Node> anonymous_object
%type <Node> anonymous_object_initialization
%type <Node> select_clause
%type <Node> group_clause
%type <Node> select_or_group_clause
%type <Node> from_clause
%type <Node> where_clause
%type <Node> query_body_clauses
%type <Node> opt_query_body_clauses
%type <Node> query_body_clause
%type <Node> let_clause
%type <Node> query_body
%type <Node> query_continuation
%type <Node> invocation_expression
%type <Node> opt_argument_list
%type <Node> argument_list
%type <Node> join_clause
%type <Node> join_into_clause
%type <Node> identifier
%type <Node> orderby_clause
%type <Node> orderings
%type <Node> ordering
%type <Ordering> ordering_direction

%%

translation_unit : expressions ;

expressions 
    : query 
    | expressions query
    ;

query
	: { fend_query_init(); } query_expression SEMICOLON { fend_query_cleanup($2); }
	;
	
query_expression
	: from_clause query_body { $$ = fend_query_complete($1, $2); }
	;

query_body
	: opt_query_body_clauses select_or_group_clause query_continuation { $$ = fend_on_query_body($1, $2, $3); }
	;

opt_query_body_clauses
	: { $$ = NULL; }
	| query_body_clauses { $$ = $1; }
	;

query_continuation
	: INTO identifier query_body { $$ = fend_on_continuation($2, $3); } %dprec 2
	| { $$ = NULL; } %dprec 1
	;

query_body_clauses
	: query_body_clause { $$ = $1; }
	| query_body_clauses query_body_clause { $$ = fend_on_enum($1, $2); }
	;

query_body_clause
	: from_clause
	| let_clause
    | where_clause
    | join_clause
    | join_into_clause
    | orderby_clause
	;

from_clause
	: FROM typedef WITHIN expression { $$ = fend_on_from($2, $4); }
	;

let_clause
	: LET identifier ASSIGN expression { $$ = fend_on_let($2, $4); }
	;

where_clause
	: WHERE boolean_expression { $$ = fend_on_where($2); }
	;

join_clause
	: JOIN typedef WITHIN expression ON expression EQUALS expression { $$ = fend_on_join($2, $4, $6, $8); }
	;

join_into_clause
	: JOIN typedef WITHIN expression ON expression EQUALS expression INTO identifier {$$ = fend_on_continuation($10, fend_on_join($2, $4, $6, $8));}
	;

orderby_clause
	: ORDERBY orderings { $$ = fend_on_order_by($2); }
	;

orderings
	: ordering { $$ = $1; }
	| orderings COMMA ordering { $$ = fend_on_enum($1, $3); }
	;

ordering
	: expression ordering_direction { $$ = fend_on_ordering($1, $2); }
	;

ordering_direction
	: { $$ = OrderingAsc; }
	| ASCENDING { $$ = $1; }
	| DESCENDING { $$ = $1; }
	;

select_or_group_clause
	: select_clause
	| group_clause
	;

select_clause
	: SELECT expression { $$ = $2; }
	;

group_clause
	: GRP expression BY expression { $$ = fend_on_group($2, $4); }
	;

boolean_expression
	: conditional_or_expression
	;

conditional_or_expression
	: conditional_and_expression { $$ = $1; }
	| conditional_or_expression OR conditional_and_expression { $$ = fend_on_predicate($1, $3, NodeTypeOrRel); }
	;

conditional_and_expression
	: not_expression { $$ = $1; }
	| conditional_and_expression AND not_expression { $$ = fend_on_predicate($1, $3, NodeTypeAndRel); }
	;

not_expression
	: exclusive_or_expression { $$ = $1; }
	| NOT exclusive_or_expression { $$ = fend_on_predicate($2, NULL, NodeTypeNotRel); }
	;

exclusive_or_expression
	: OPEN_PAREN boolean_expression CLOSE_PAREN { $$ = $2; }
	| relational_expr { $$ = $1; }
	;

expression
	: unary_expression
	| anonymous_object
	| query_expression
	;
	
unary_expression
	: identifier { $$ = fend_on_unary_expression(UnaryExpTypeIdentifier, $1, NULL); }
	| identifier DOT attribute { $$ = fend_on_unary_expression(UnaryExpTypePropertyCall, $1, $3); }
	| identifier DOT invocation_expression { $$ = fend_on_unary_expression(UnaryExpTypeMehtodCall, $1, $3); }
	| STRING { $$ = fend_on_unary_expression(UnaryExpTypeString, $1, NULL); }
	| INTEGER { $$ = fend_on_unary_expression(UnaryExpTypeNumber, $1, NULL); }
	;
	
anonymous_object
	: OPEN_BRACE anonymous_object_initialization CLOSE_BRACE { $$ = $2; }
	;
	
anonymous_object_initialization
	: unary_expression { $$ = $1; }
	| anonymous_object_initialization COMMA unary_expression { $$ = fend_on_enum($1, $3); }
	;

relational_expr
	: unary_expression REL_OP unary_expression { $$ = fend_on_releational_expr($1, $3, $2); } // TODO: implement type checking
	;

identifier 
	: IDENTIFIER { $$ = fend_on_identifier($1); }
	;
	
attribute
	: IDENTIFIER { $$ = fend_on_string_attribute($1); }
	| TYPE { $$ = fend_on_type_attribute($1); }
	;

invocation_expression 
	: IDENTIFIER OPEN_PAREN opt_argument_list CLOSE_PAREN {$$ = fend_on_method_call($1, $3); }
	;

opt_argument_list
	: argument_list { $$ = $1; }
	| { $$ = NULL; }
	;

argument_list
	: expression { $$ = $1; }
	| argument_list COMMA expression { $$ = fend_on_enum($1, $3); }
	;

typedef
    : type identifier { $$ = fend_on_identifier_declaration($1, $2); }
	| identifier { $$ = fend_on_identifier_declaration(fend_on_simple_type_def(TypeDefDynamic), $1); }
	;

type
	: TYPE { $$ = $1; }
	| IDENTIFIER { $$ = fend_on_complex_type_def(TypeDefUser, $1); }
	;

%%

int yyerror(char* s) {
	lib_fprintf(stderr, "%d: %s at %s\n", yylineno, s, yytext);
	fend_query_cleanup(NULL);
	return 1;
}
from pydantic import AliasGenerator, ConfigDict
from pydantic.alias_generators import to_camel

# Contratos orientados a interoperar con el backend .NET (camelCase en HTTP).
SALIDA_CFG = ConfigDict(alias_generator=AliasGenerator(serialization_alias=to_camel), populate_by_name=True)

ENTRADA_CFG = ConfigDict(
    alias_generator=AliasGenerator(validation_alias=to_camel),
    populate_by_name=True,
    extra="forbid",
)

import enum
import logging
from abc import ABCMeta
from typing import Generic, Optional, Type, TypeVar

from pluggy import PluginManager
from pydantic import BaseModel

from docling.datamodel.pipeline_options import BaseOptions
from docling.models.base_model import BaseModelWithOptions

A = TypeVar("A", bound=BaseModelWithOptions)


logger = logging.getLogger(__name__)


class FactoryMeta(BaseModel):
    """A Pydantic model for storing metadata about a registered factory item.

    Attributes:
        kind: The kind of the registered item (e.g., "easyocr").
        plugin_name: The name of the plugin that registered the item.
        module: The name of the module where the item is defined.
    """

    kind: str
    plugin_name: str
    module: str


class BaseFactory(Generic[A], metaclass=ABCMeta):
    """A generic factory for creating instances of models from plugins.

    This class provides a framework for discovering, registering, and creating
    instances of models (or other classes) that are defined in external plugins.
    It uses `pluggy` to load plugins from setuptools entry points.

    TypeVar:
        A: The type of the objects that this factory creates.
    """

    default_plugin_name = "docling"

    def __init__(self, plugin_attr_name: str, plugin_name=default_plugin_name):
        """Initializes the BaseFactory.

        Args:
            plugin_attr_name: The name of the attribute to look for in plugin
                modules to get the configuration.
            plugin_name: The name of the plugin group to load.
        """
        self.plugin_name = plugin_name
        self.plugin_attr_name = plugin_attr_name

        self._classes: dict[Type[BaseOptions], Type[A]] = {}
        self._meta: dict[Type[BaseOptions], FactoryMeta] = {}

    @property
    def registered_kind(self) -> list[str]:
        """Returns a list of the 'kind' strings for all registered classes."""
        return [opt.kind for opt in self._classes.keys()]

    def get_enum(self) -> enum.Enum:
        """Creates an `Enum` of all registered kinds."""
        return enum.Enum(
            self.plugin_attr_name + "_enum",
            names={kind: kind for kind in self.registered_kind},
            type=str,
            module=__name__,
        )

    @property
    def classes(self):
        """Returns a dictionary of the registered classes."""
        return self._classes

    @property
    def registered_meta(self):
        """Returns a dictionary of the metadata for all registered classes."""
        return self._meta

    def create_instance(self, options: BaseOptions, **kwargs) -> A:
        """Creates an instance of a registered class based on the provided options.

        Args:
            options: The options object that determines which class to instantiate.
            **kwargs: Additional keyword arguments to pass to the class constructor.

        Returns:
            An instance of the requested class.

        Raises:
            RuntimeError: If no class is found for the given options type.
        """
        try:
            _cls = self._classes[type(options)]
            return _cls(options=options, **kwargs)
        except KeyError:
            raise RuntimeError(self._err_msg_on_class_not_found(options.kind))

    def create_options(self, kind: str, *args, **kwargs) -> BaseOptions:
        """Creates an options object for a given kind.

        Args:
            kind: The 'kind' string of the options object to create.
            *args: Positional arguments to pass to the options constructor.
            **kwargs: Keyword arguments to pass to the options constructor.

        Returns:
            An instance of the requested options class.

        Raises:
            RuntimeError: If no options class is found for the given kind.
        """
        for opt_cls, _ in self._classes.items():
            if opt_cls.kind == kind:
                return opt_cls(*args, **kwargs)
        raise RuntimeError(self._err_msg_on_class_not_found(kind))

    def _err_msg_on_class_not_found(self, kind: str):
        msg = []

        for opt, cls in self._classes.items():
            msg.append(f"\t{opt.kind!r} => {cls!r}")

        msg_str = "\n".join(msg)

        return f"No class found with the name {kind!r}, known classes are:\n{msg_str}"

    def register(self, cls: Type[A], plugin_name: str, plugin_module_name: str):
        """Registers a class with the factory.

        Args:
            cls: The class to register.
            plugin_name: The name of the plugin that provides the class.
            plugin_module_name: The name of the module where the class is defined.

        Raises:
            ValueError: If a class with the same options type is already registered.
        """
        opt_type = cls.get_options_type()

        if opt_type in self._classes:
            raise ValueError(
                f"{opt_type.kind!r} already registered to class {self._classes[opt_type]!r}"
            )

        self._classes[opt_type] = cls
        self._meta[opt_type] = FactoryMeta(
            kind=opt_type.kind, plugin_name=plugin_name, module=plugin_module_name
        )

    def load_from_plugins(
        self, plugin_name: Optional[str] = None, allow_external_plugins: bool = False
    ):
        """Loads and registers classes from plugins.

        This method uses `pluggy` to discover and load plugins from setuptools
        entry points. It then calls `process_plugin` to register the classes
        provided by each plugin.

        Args:
            plugin_name: The name of the plugin group to load.
            allow_external_plugins: If `False`, only plugins from the `docling`
                namespace will be loaded.
        """
        plugin_name = plugin_name or self.plugin_name

        plugin_manager = PluginManager(plugin_name)
        plugin_manager.load_setuptools_entrypoints(plugin_name)

        for plugin_name, plugin_module in plugin_manager.list_name_plugin():
            plugin_module_name = str(plugin_module.__name__)  # type: ignore

            if not allow_external_plugins and not plugin_module_name.startswith(
                "docling."
            ):
                logger.warning(
                    f"The plugin {plugin_name} will not be loaded because Docling is being executed with allow_external_plugins=false."
                )
                continue

            attr = getattr(plugin_module, self.plugin_attr_name, None)

            if callable(attr):
                logger.info("Loading plugin %r", plugin_name)

                config = attr()
                self.process_plugin(config, plugin_name, plugin_module_name)

    def process_plugin(self, config, plugin_name: str, plugin_module_name: str):
        """Processes a plugin's configuration and registers its classes.

        Args:
            config: The configuration object returned by the plugin.
            plugin_name: The name of the plugin.
            plugin_module_name: The name of the plugin's module.
        """
        for item in config[self.plugin_attr_name]:
            try:
                self.register(item, plugin_name, plugin_module_name)
            except ValueError:
                logger.warning("%r already registered", item)

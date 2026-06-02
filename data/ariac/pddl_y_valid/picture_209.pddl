(define (problem picture_209)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
         green_pump blue_regulator red_battery - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear green_pump)
        (clear blue_regulator)
        (clear red_battery)
        (part_at green_pump table)
        (part_at blue_regulator table)
        (part_at red_battery table)
    )
    
    (:goal
        (and
            (part_at green_pump pump_placement)
            (part_at blue_regulator regulator_placement)
            (part_at red_battery battery_placement)
        )
    )
)